//! Synthetic GFX10 compute programs and an independent lane-value oracle.

use super::{Lower, Lowered, Parts, lower};
use crate::emit::bitcode::wave_eval;
use crate::ir::Reg;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const OLD: u32 = 0xdead_beef;

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Swap16,
    Xor1,
    Quad,
    Identity,
}

#[derive(Clone, Copy, Debug)]
enum Exec {
    Full,
    Edges,
    Nested,
    Empty,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    wave: u32,
    threads: u32,
    pattern: Pattern,
    exec: Exec,
    seed: u32,
}

fn cases() -> impl Iterator<Item = Case> {
    [(32, 32), (32, 16), (32, 23), (64, 48), (64, 64)]
        .into_iter()
        .flat_map(|(wave, threads)| {
            [
                Pattern::Xor1,
                Pattern::Swap16,
                Pattern::Quad,
                Pattern::Identity,
            ]
            .into_iter()
            .flat_map(move |pattern| {
                [Exec::Edges, Exec::Nested, Exec::Empty, Exec::Full]
                    .into_iter()
                    .zip([0x7f80_0000, 0xff80_0000, 17, u32::MAX])
                    .map(move |(exec, seed)| Case {
                        wave,
                        threads,
                        pattern,
                        exec,
                        seed,
                    })
            })
        })
}

/// Reference behavior in AMD RDNA2 ISA §12.13.1. Use explicit lane sets
/// and permutations independent of the fixture instruction encodings.
fn expected(case: Case, lane: u32) -> u32 {
    let enabled = |lane: u32| {
        lane < case.threads
            && match case.exec {
                Exec::Full => true,
                Exec::Edges => [0, 15, 16, 31, 32, 47, 48, 63].contains(&lane),
                Exec::Nested => [0, 15, 32, 47].contains(&lane),
                Exec::Empty => false,
            }
    };
    if !enabled(lane) {
        return OLD;
    }
    let source = match case.pattern {
        Pattern::Swap16 => lane / 32 * 32 + (lane + 16) % 32,
        Pattern::Xor1 => [1, 0][lane as usize % 2] + lane / 2 * 2,
        Pattern::Quad => [1, 0, 3, 2][lane as usize % 4] + lane / 4 * 4,
        Pattern::Identity => lane,
    };
    if enabled(source) {
        case.seed.wrapping_add(source)
    } else {
        0
    }
}

fn vop2(op: u32, dst: u32, src0: u32, src1: u32) -> u32 {
    op << 25 | dst << 17 | src1 << 9 | src0
}

fn vopc(op: u32, src0: u32, src1: u32) -> u32 {
    0x7c00_0000 | op << 17 | src1 << 9 | src0
}

fn sop1(op: u32, dst: u32, src: u32) -> u32 {
    0xbe80_0000 | dst << 16 | op << 8 | src
}

fn target(code: &mut [u32], branch: usize, to: usize) {
    let delta = i16::try_from(to as isize - branch as isize - 1).unwrap();
    code[branch] |= u32::from(delta as u16);
}

fn fixture(case: Case, flow: bool) -> NativeProgram {
    let wave64 = case.wave == 64;
    let save = if wave64 { 0x24 } else { 0x3c }; // s_and_saveexec
    let restore = if wave64 { 0x04 } else { 0x03 }; // s_mov
    let mut code = vec![
        vop2(0x25, 1, 255, 0),
        case.seed, // v1 = v0 + seed, modulo 2^32
        0x7e04_02ff,
        OLD, // v_mov_b32 v2, OLD
    ];
    if flow {
        code.push(sop1(3, 12, 130)); // s_mov_b32 s12, 2
    }
    let header = code.len();
    if flow {
        code.push(vop2(0x25, 1, 129, 1)); // v_add_nc_u32 v1, 1, v1
    }
    let mut outer = None;
    let mut inner = None;
    match case.exec {
        Exec::Edges | Exec::Nested => {
            code.extend([
                vop2(0x25, 3, 129, 0), // v3 = lane + 1
                vop2(0x16, 3, 129, 3), // v3 >>= 1
                vop2(0x1b, 3, 135, 3), // v3 &= 7
                vopc(0xc2, 128, 3),    // v_cmp_eq_u32 0, v3
                sop1(save, 8, 106),    // s_and_saveexec s8, vcc
            ]);
            if flow {
                outer = Some(code.len());
                code.push(0xbf88_0000); // s_cbranch_execz to outer EXEC restore
            }
            if matches!(case.exec, Exec::Nested) {
                code.extend([
                    vop2(0x1b, 3, 144, 0), // v3 = lane & 16
                    vopc(0xc2, 128, 3),    // v_cmp_eq_u32 0, v3
                    sop1(save, 10, 106),   // s_and_saveexec s10, vcc
                ]);
                if flow {
                    inner = Some(code.len());
                    code.push(0xbf88_0000); // s_cbranch_execz to inner EXEC restore
                }
            }
        }
        Exec::Empty => code.extend([
            vopc(0xc1, 192, 0), // v_cmp_lt_u32 64, v0
            sop1(save, 8, 106),
        ]),
        Exec::Full => {}
    }
    let offset = match case.pattern {
        Pattern::Swap16 => 0x401f,
        Pattern::Xor1 => 0x041f,
        Pattern::Quad => 0x80b1,
        Pattern::Identity => 0x001f,
    };
    code.extend([0xd8d4_0000 | offset, 0x0200_0001]); // ds_swizzle_b32 v2, v1
    if matches!(case.exec, Exec::Nested) {
        if let Some(branch) = inner {
            let to = code.len();
            target(&mut code, branch, to);
        }
        code.push(sop1(restore, 126, 10));
    }
    if !matches!(case.exec, Exec::Full) {
        if let Some(branch) = outer {
            let to = code.len();
            target(&mut code, branch, to);
        }
        code.push(sop1(restore, 126, 8));
    }
    if flow {
        code.extend([0x808c_810c, 0xbf07_800c]); // s12 -= 1; s_cmp_lg_u32 s12, 0
        let latch = code.len();
        code.push(0xbf85_0000); // s_cbranch_scc1 to loop header
        target(&mut code, latch, header);
    }
    code.push(vop2(0x1a, 4, 130, 0)); // v4 = lane * 4
    if flow {
        code.extend([
            0xd834_0000,
            0x0000_0204, // ds_write_b32 v4, v2
            0xbf8c_0000, // s_waitcnt 0
            0xbf8a_0000, // s_barrier after reconvergence, all 64 threads
            vop2(0x1d, 3, 255, 4),
            128, // v3 = peer's byte address, in the other wave32
            0xd8d8_0000,
            0x0200_0003, // ds_read_b32 v2, v3
            0xbf8c_0000, // s_waitcnt 0
        ]);
    }
    code.extend([
        sop1(3, 1, 128), // s_mov_b32 s1, 0
        0xf408_0100,
        0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
        0xe070_1000,
        0x8001_0204, // buffer_store_dword v2, v4, s[4:7], 0 offen
        0xbf81_0000, // s_endpgm
    ]);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 5,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 | if flow { 1 << 15 } else { 0 },
            threads: [case.threads, 1, 1],
        }),
        code: code.into_iter().flat_map(u32::to_le_bytes).collect(),
    }
}

fn with_parts<T>(case: Case, flow: bool, test: impl FnOnce(&Parts<'_>) -> T) -> T {
    let native = fixture(case, flow);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("decode GFX10 fixture");
    let res = resource::track(&p, &native.layout, native.stage).expect("track fixture UAV");
    let io = io::raise(&p, &native).expect("raise fixture IO");
    let st = structure::structure(&p).expect("structure fixture");
    test(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

#[test]
fn ds_swizzle_exec_cpu_values() {
    for case in cases() {
        with_parts(case, false, |parts| {
            let bad = Default::default();
            let mut lower = Lower::new(parts, &bad);
            assert!(lower.inputs().is_ok());
            let [block] = parts.p.blocks.as_slice() else {
                panic!("straight-line fixture")
            };
            for (i, inst) in block.insts.iter().enumerate() {
                assert!(lower.inst(0, i).is_ok(), "{case:?}: {}", inst.name());
                if inst.name() != "ds_swizzle_b32" {
                    continue;
                }
                let value = lower.i32_of(inst.write(Reg::V(2)).unwrap()).unwrap();
                let values = wave_eval::evaluate(&lower.s.m, value, case.threads);
                for (lane, got) in values.into_iter().enumerate() {
                    assert_eq!(got, expected(case, lane as u32), "{case:?}, lane {lane}");
                }
                return;
            }
            panic!("fixture did not decode a swizzle");
        });
    }
}

fn gpu_cases() -> impl Iterator<Item = (Case, bool)> {
    cases()
        .map(|case| (case, false))
        .chain([Exec::Edges, Exec::Nested].into_iter().map(|exec| {
            (
                Case {
                    wave: 32,
                    threads: 64,
                    pattern: Pattern::Swap16,
                    exec,
                    seed: 0x7f80_0000,
                },
                true,
            )
        }))
}

fn compiled(case: Case, flow: bool) -> Lowered {
    with_parts(case, flow, |parts| {
        lower(parts).unwrap_or_else(|e| panic!("{case:?}, flow={flow}: {e}"))
    })
}

#[test]
fn ds_swizzle_exec_dxil_fixtures() {
    let dump = std::env::var_os("VICTORY_TEST_SWIZZLE_DXIL_DIR").map(std::path::PathBuf::from);
    for (id, (case, flow)) in gpu_cases().enumerate() {
        let shader = compiled(case, flow);
        if let Some(dir) = &dump {
            std::fs::write(dir.join(format!("swizzle-{id:02}.dxil")), shader.dxil)
                .expect("write fixture to existing dump directory");
        }
    }
}

#[cfg(windows)]
mod gpu {
    use super::*;
    use crate::emit::HostShader;
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::validator::Validator;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;

    fn adapter_name(desc: &DXGI_ADAPTER_DESC1) -> String {
        String::from_utf16_lossy(&desc.Description)
            .trim_end_matches('\0')
            .to_owned()
    }

    /// Use WARP by default. An explicit name selects physical hardware and
    /// fails if it is missing, so a GPU run cannot silently test WARP.
    fn device() -> (Host, D3D12_FEATURE_DATA_D3D12_OPTIONS1) {
        // SAFETY: COM methods receive live interfaces and correctly sized
        // output buffers. The selected adapter is kept alive through creation.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let host = match std::env::var("VICTORY_TEST_ADAPTER") {
                Ok(selector) => {
                    let selector = selector.trim().to_ascii_lowercase();
                    assert!(!selector.is_empty(), "empty VICTORY_TEST_ADAPTER");
                    let adapter = (0..)
                        .map_while(|i| match factory.EnumAdapters1(i) {
                            Ok(adapter) => Some(adapter),
                            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => None,
                            Err(e) => panic!("enumerate adapters: {e}"),
                        })
                        .find(|adapter| {
                            let desc = adapter.GetDesc1().unwrap();
                            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0
                                && adapter_name(&desc).to_ascii_lowercase().contains(&selector)
                        })
                        .unwrap_or_else(|| panic!("no physical adapter matching {selector:?}"));
                    let mut device = None;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).unwrap();
                    Host {
                        device: device.unwrap(),
                    }
                }
                Err(std::env::VarError::NotPresent) => {
                    Host::new(true).expect("create WARP adapter")
                }
                Err(e) => panic!("read VICTORY_TEST_ADAPTER: {e}"),
            };
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapterByLuid(host.device.GetAdapterLuid())
                .unwrap();
            let desc = adapter.GetDesc1().unwrap();
            let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            host.device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS1,
                    (&raw mut caps).cast(),
                    size_of_val(&caps) as u32,
                )
                .unwrap();
            eprintln!(
                "adapter={} pci={:04x}:{:04x} wave_ops={} wave_range={}..={}",
                adapter_name(&desc),
                desc.VendorId,
                desc.DeviceId,
                caps.WaveOps.as_bool(),
                caps.WaveLaneCountMin,
                caps.WaveLaneCountMax,
            );
            (host, caps)
        }
    }

    #[test]
    #[ignore = "requires D3D12 and an SM 6.6 adapter; run explicitly"]
    fn ds_swizzle_exec_gpu() {
        let (host, caps) = device();
        assert!(
            caps.WaveOps.as_bool(),
            "adapter does not support wave operations"
        );
        let mut validator = Validator::load(Validator::default_path().as_deref());
        if let Err(e) = &validator {
            assert!(
                std::env::var_os("VICTORY_DXIL").is_none(),
                "explicit VICTORY_DXIL failed to load: {e}"
            );
            eprintln!("optional DXIL validator unavailable: {e}");
        }
        let mut ran = 0;
        let mut skipped = 0;
        for (id, (case, flow)) in gpu_cases().enumerate() {
            if !(caps.WaveLaneCountMin..=caps.WaveLaneCountMax).contains(&case.wave) {
                skipped += 1;
                continue;
            }
            eprintln!("dispatch {id:02}: {case:?}, flow={flow}");
            let output = compiled(case, flow);
            let mut shader = HostShader {
                stage: HwStage::Cs,
                dxil: output.dxil,
                user_data: output.user_data,
                samplers: output.samplers,
                static_samplers: vec![],
                root_cbvs: output.root_cbvs,
                root_srvs: output.root_srvs,
            };
            if let Ok(v) = &mut validator {
                v.validate(&mut shader.dxil)
                    .expect("DXIL validator rejected fixture");
            }
            let (root, pso) = host.compute_pipeline(&shader).expect("create compute PSO");
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(
                            0,
                            Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                        )],
                        views: &[View::RawUav(case.threads)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch and read back raw UAV");
            let [values] = out.as_slice() else {
                panic!("expected one raw UAV")
            };
            assert_eq!(values.len(), case.threads as usize);
            for (lane, &got) in values.iter().enumerate() {
                let (reference_case, source_lane) = if flow {
                    (
                        Case {
                            seed: case.seed.wrapping_add(2),
                            ..case
                        },
                        lane as u32 ^ 32,
                    )
                } else {
                    (case, lane as u32)
                };
                assert_eq!(
                    got,
                    expected(reference_case, source_lane),
                    "{case:?}, flow={flow}, lane {lane}"
                );
            }
            ran += 1;
        }
        eprintln!("fixtures passed={ran}, skipped (unsupported wave size)={skipped}");
        assert!(ran > 0, "no fixture ran on the selected adapter");
    }
}
