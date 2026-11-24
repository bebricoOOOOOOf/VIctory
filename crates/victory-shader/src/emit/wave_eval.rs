//! Evaluates the DXIL value graph used by the single-wave swizzle fixtures.
//! A call to WaveReadLaneAt targeting an absent host lane fails the test.

use super::{Bin, Const, Module, Op, V, pred};
use std::collections::HashMap;

pub(crate) fn evaluate(module: &Module, value: V, threads: u32) -> Vec<u32> {
    let mut wave = Wave {
        module,
        threads,
        cache: HashMap::new(),
    };
    for (id, inst) in module.insts.iter().enumerate() {
        if let Op::Call(f, _) = inst.op
            && module.funcs[f as usize].name == "dx.op.waveReadLaneAt.i32"
        {
            // The call runs on every thread at the swizzle site, including
            // those whose destination write is later suppressed by EXEC.
            for lane in 0..threads {
                wave.eval(V::Inst(id as u32), lane);
            }
        }
    }
    (0..threads).map(|lane| wave.eval(value, lane)[0]).collect()
}

struct Wave<'a> {
    module: &'a Module,
    threads: u32,
    cache: HashMap<(V, u32), [u32; 4]>,
}

impl Wave<'_> {
    fn eval(&mut self, value: V, lane: u32) -> [u32; 4] {
        assert!(
            lane < self.threads,
            "DXIL read of inactive host lane {lane}"
        );
        if let Some(&v) = self.cache.get(&(value, lane)) {
            return v;
        }
        let scalar = |v| [v, 0, 0, 0];
        let module = self.module;
        let mut get = |v| self.eval(v, lane)[0];
        let result = match value {
            V::Const(i) => match module.consts[i as usize].1 {
                Const::Int(v) => scalar(v as u32),
                ref other => panic!("unexpected fixture constant {other:?}"),
            },
            V::Inst(i) => match &module.insts[i as usize].op {
                Op::Bin(op, a, b) => {
                    let (a, b) = (get(*a), get(*b));
                    scalar(match op {
                        Bin::Add => a.wrapping_add(b),
                        Bin::Sub => a.wrapping_sub(b),
                        Bin::URem => a % b,
                        Bin::Shl => a << b,
                        Bin::LShr => a >> b,
                        Bin::And => a & b,
                        Bin::Or => a | b,
                        Bin::Xor => a ^ b,
                        other => panic!("unexpected fixture binop {other:?}"),
                    })
                }
                Op::Cmp(op, a, b) => {
                    let (a, b) = (get(*a), get(*b));
                    scalar(u32::from(match *op {
                        pred::EQ => a == b,
                        pred::NE => a != b,
                        pred::IULT => a < b,
                        other => panic!("unexpected fixture comparison {other}"),
                    }))
                }
                Op::Select(c, t, f) => scalar(if get(*c) != 0 { get(*t) } else { get(*f) }),
                Op::Extract(v, index) => scalar(self.eval(*v, lane)[*index as usize]),
                Op::Call(_, args) => match get(args[0]) {
                    96 | 111 => scalar(lane), // The fixture starts at group index zero.
                    116 => {
                        let mut ballot = [0; 4];
                        for voter in 0..self.threads {
                            if self.eval(args[1], voter)[0] != 0 {
                                ballot[voter as usize / 32] |= 1 << (voter % 32);
                            }
                        }
                        ballot
                    }
                    117 => {
                        let source = get(args[2]);
                        self.eval(args[1], source)
                    }
                    118 => self.eval(args[1], 0),
                    other => panic!("unexpected fixture DXIL opcode {other}"),
                },
                other => panic!("unexpected fixture operation {other:?}"),
            },
            other => panic!("unexpected fixture value {other:?}"),
        };
        self.cache.insert((value, lane), result);
        result
    }
}
