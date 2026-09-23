// SPDX-License-Identifier: MIT
use crate::{
    isa::*,
    mir::{self, Op},
    schedule,
};
#[test]
fn encoding() {
    for &(op, _) in NAMES {
        let mut i = Inst::new(op);
        if op == 3 {
            i.imm = 1;
        }
        assert_eq!(Inst::decode(i.encode().unwrap()).unwrap(), i);
        assert_eq!(assemble(&disassemble(&[i])).unwrap(), vec![i]);
    }
    assert!(Inst::decode(0xff).is_err());
    assert!(Inst::decode(1 << 32).is_err());
    assert!(Inst {
        d: 63,
        ..Inst::new(0x2c)
    }
    .encode()
    .is_err());
}
#[test]
fn raw_waw_tokens_and_branches() {
    let input = [
        Inst {
            d: 1,
            a: 2,
            b: 3,
            ..Inst::new(0x24)
        },
        Inst {
            d: 1,
            ..Inst::new(0x20)
        },
        Inst {
            d: 4,
            a: 8,
            ..Inst::new(0x50)
        },
        Inst {
            d: 5,
            a: 4,
            b: 1,
            ..Inst::new(0x22)
        },
        Inst {
            a: 8,
            b: 5,
            ..Inst::new(0x51)
        },
        Inst {
            imm: 6,
            ..Inst::new(4)
        },
        Inst::new(1),
    ];
    let code = schedule::schedule(&input).unwrap();
    schedule::validate(&code).unwrap();
    assert!(code.len() > input.len());
    let b = code.iter().find(|i| i.op == 4).unwrap();
    assert_eq!(code[b.imm as usize].op, 1);
    assert!(schedule::validate(&input).is_err());
    let mut loads: Vec<_> = (0..9)
        .map(|d| Inst {
            d,
            a: 40,
            ..Inst::new(0x50)
        })
        .collect();
    loads.push(Inst::new(1));
    let code = schedule::schedule(&loads).unwrap();
    schedule::validate(&code).unwrap();
    assert_eq!(code[4].op, 3);
    assert_eq!(code[4].imm, 1);
}
#[test]
fn fence_drain() {
    for boundary in [7, 8] {
        // A fence cannot pass a live token even when it writes no register.
        let input = [
            Inst {
                a: 2,
                b: 4,
                ..Inst::new(0x51)
            },
            Inst::new(boundary),
            Inst::new(1),
        ];
        assert!(schedule::validate(&input).is_err());
        let code = schedule::schedule(&input).unwrap();
        assert_eq!(
            code[1],
            Inst {
                imm: 1,
                ..Inst::new(3)
            }
        );
        assert_eq!(code[2], Inst::new(boundary));
        schedule::validate(&code).unwrap();

        // No operand dependency connects the FP result to the fence.
        let input = [Inst::new(0x30), Inst::new(boundary), Inst::new(1)];
        assert!(schedule::validate(&input).is_err());
        let code = schedule::schedule(&input).unwrap();
        assert_eq!(code[FP32_LATENCY], Inst::new(boundary));
        schedule::validate(&code).unwrap();
        assert!(Inst {
            imm: 1,
            ..Inst::new(boundary)
        }
        .validate()
        .is_err());
        assert!(Inst {
            a: 1,
            ..Inst::new(boundary)
        }
        .validate()
        .is_err());
    }
}
#[test]
fn metadata_and_private_bounds() {
    let p = mir::compile(&[Op::new(0x40, 0, 0, 0, 0, 0)], 0, 0).unwrap();
    let mut bytes = p.bytes().unwrap();
    assert_eq!(Program::parse(&bytes).unwrap().code, p.code);
    bytes.push(0);
    assert!(Program::parse(&bytes).is_err());
    assert_eq!(
        private_address(4096, 2, 80, 1, 2, 15, 10240).unwrap(),
        4096 + (82 * 16 + 15) * 4
    );
    assert!(private_address(u64::MAX - 8, 1, 1, 0, 0, 0, 64).is_err());
    assert!(private_address(0, u64::MAX, 80, 0, 0, 0, u64::MAX).is_err());
    assert!(private_address(0, 1, 80, 0, 80, 0, 5120).is_err());
}
#[test]
fn pressure_and_long_chain() {
    let mut chain = vec![Op::new(0x40, 0, 0, 0, 0, 0), Op::new(0x20, 1, 0, 0, 0, 3)];
    let mut v = 0;
    for d in 2..152 {
        chain.push(Op::new(0x22, d, v, 1, 0, 0));
        v = d;
    }
    chain.extend([
        Op::new(0x20, 500, 0, 0, 0, 0),
        Op::new(0x54, 0, 500, v, 0, 0),
    ]);
    let p = mir::compile(&chain, 4, 0).unwrap();
    assert_eq!(p.private, 0);
    for lane in 0..16 {
        assert_eq!(arithmetic_slice(&p, lane), lane + 450);
    }
    let mut pressure = vec![Op::new(0x40, 0, 0, 0, 0, 0)];
    // Exact legacy semantic input: 80 live lane << (index % 8) values.
    for id in 1..=80 {
        pressure.push(Op::new(0x20, id, 0, 0, 0, (id - 1) % 8));
        pressure.push(Op::new(0x28, id + 100, 0, id, 0, 0));
    }
    pressure.push(Op::new(0x20, 200, 0, 0, 0, 0));
    let mut sum = 200;
    for id in 1..=80 {
        pressure.push(Op::new(0x22, 200 + id, sum, 100 + id, 0, 0));
        sum = 200 + id;
    }
    pressure.extend([
        Op::new(0x20, 500, 0, 0, 0, 0),
        Op::new(0x54, 0, 500, sum, 0, 0),
    ]);
    let p = mir::compile(&pressure, 4, 0).unwrap();
    assert!(p.private > 0);
    assert!(p.vector <= 64);
    schedule::validate(&p.code).unwrap();
    for lane in 0..16 {
        assert_eq!(arithmetic_slice(&p, lane), lane * 2550);
    }
    let reserved = mir::compile(&pressure, 4, 64).unwrap();
    assert_eq!(reserved.private, p.private + 64);
    assert!(reserved
        .code
        .iter()
        .filter(|i| matches!(i.op, 0x56 | 0x57))
        .all(|i| i.imm >= 16));
    for lane in 0..16 {
        assert_eq!(arithmetic_slice(&reserved, lane), lane * 2550);
    }
}

#[test]
fn scalar_control_lifetimes() {
    fn evaluate(ops: &[Op]) -> u32 {
        let program = mir::compile(ops, 0, 0).unwrap();
        schedule::validate(&program.code).unwrap();
        let mut s = [0u32; 64];
        let mut mask = 0xffff;
        let mut pc = 0;
        for _ in 0..10000 {
            let i = program.code[pc];
            pc += 1;
            let a = s[i.a as usize];
            let b = s[i.b as usize];
            let result = match i.op {
                0 | 3 => continue,
                4 => {
                    pc = i.imm as usize;
                    continue;
                }
                5 => {
                    if a != 0 {
                        pc = i.imm as usize;
                    }
                    continue;
                }
                6 => {
                    let old = mask;
                    mask = a & 0xffff;
                    old
                }
                0x10 => i.imm,
                0x11 => a,
                0x12 => a.wrapping_add(b),
                0x13 => a.wrapping_sub(b),
                0x2d => {
                    assert_eq!(mask, 0xffff);
                    return a;
                }
                op => panic!("outside scalar allocation slice: {op:x}"),
            };
            s[i.d as usize] = result;
        }
        panic!("scalar allocation slice did not terminate");
    }
    let mut sequential = vec![Op::new(0x10, 0, 0, 0, 0, 0)];
    for id in 1..=80 {
        sequential.extend([
            Op::new(0x10, id, 0, 0, 0, id),
            Op::new(6, 100 + id, id, 0, 0, 0),
            Op::new(0x12, 0, 0, id, 0, 0),
            Op::new(6, 200 + id, 100 + id, 0, 0, 0),
        ]);
    }
    sequential.push(Op::new(0x2d, 999, 0, 0, 0, 0));
    assert_eq!(evaluate(&sequential), 80 * 81 / 2);

    // The invariant's final textual use precedes later loop temporaries.
    // Its register must survive both inner and outer backedges.
    let nested = [
        Op::new(0x10, 0, 0, 0, 0, 0), // sum
        Op::new(0x10, 1, 0, 0, 0, 7), // invariant
        Op::new(0x10, 2, 0, 0, 0, 3), // outer count
        Op::new(0x10, 3, 0, 0, 0, 1),
        Op::new(0x12, 0, 0, 1, 0, 0), // outer header
        Op::new(0x10, 5, 0, 0, 0, 2), // inner count
        Op::new(0x12, 0, 0, 2, 0, 0), // inner header
        Op::new(0x13, 5, 5, 3, 0, 0),
        Op::new(5, 0, 5, 0, 0, 6),
        Op::new(0x13, 2, 2, 3, 0, 0),
        Op::new(5, 0, 2, 0, 0, 4),
        Op::new(0x2d, 999, 0, 0, 0, 0),
    ];
    assert_eq!(evaluate(&nested), 3 * 7 + 2 * (3 + 2 + 1));
}

// Deliberately narrow compiler-only arithmetic/spill evaluator: one lane,
// straight-line integer code, no queues, MMU, cache, or RTL timing claims.
fn arithmetic_slice(p: &Program, lane: u32) -> u32 {
    let mut regs = [0u32; 64];
    let mut private = vec![0; p.private as usize / 4];
    let mut output = None;
    for i in &p.code {
        let a = regs[i.a as usize];
        let b = regs[i.b as usize];
        let value = match i.op {
            0 | 1 | 3 => continue,
            0x20 => i.imm,
            0x21 => a,
            0x22 => a.wrapping_add(b),
            0x24 => a.wrapping_mul(b),
            0x28 => a.wrapping_shl(b & 31),
            0x40 => lane,
            0x56 => private[i.imm as usize],
            0x57 => {
                private[i.imm as usize] = b;
                continue;
            }
            0x54 => {
                assert_eq!(a, 0);
                output = Some(b);
                continue;
            }
            op => panic!("outside arithmetic slice: {op:x}"),
        };
        regs[i.d as usize] = value;
    }
    output.unwrap()
}

#[test]
fn random_async_latency_and_loop_repatch() {
    let mut seed = 0xa6155500u32;
    for _ in 0..160 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let latency = (seed as usize % 256) + 1;
        let input = [
            Inst {
                d: 2,
                a: 40,
                ..Inst::new(0x50)
            },
            Inst {
                d: 3,
                a: 2,
                b: 8,
                ..Inst::new(0x22)
            },
            Inst::new(1),
        ];
        let code = schedule::schedule(&input).unwrap();
        let mut cycle = 0;
        let mut complete = 0;
        let mut result = None;
        let mut loaded = None;
        for i in code {
            match i.op {
                0x50 => complete = cycle + latency,
                3 => {
                    cycle = cycle.max(complete);
                    loaded = Some(seed);
                }
                0x22 => {
                    assert!(cycle >= complete);
                    result = Some(loaded.unwrap().wrapping_add(3));
                }
                1 => assert_eq!(result, Some(seed.wrapping_add(3))),
                _ => panic!("unexpected slice op"),
            }
            cycle += 1;
        }
    }
    let input = [
        Inst {
            d: 4,
            imm: 10,
            ..Inst::new(0x10)
        },
        Inst {
            d: 5,
            imm: 1,
            ..Inst::new(0x10)
        },
        Inst {
            d: 8,
            a: 12,
            b: 16,
            ..Inst::new(0x24)
        },
        Inst {
            d: 4,
            a: 4,
            b: 5,
            ..Inst::new(0x13)
        },
        Inst {
            a: 4,
            imm: 2,
            ..Inst::new(5)
        },
        Inst::new(1),
    ];
    let code = schedule::schedule(&input).unwrap();
    schedule::validate(&code).unwrap();
    let target = code.iter().find(|i| i.op == 5).unwrap().imm as usize;
    assert_eq!(code[target].op, 0x24);
    let p = Program {
        code,
        entry: 0,
        scalar: 6,
        vector: 17,
        shared: 0,
        private: 0,
    };
    assert!(Program::parse(&p.bytes().unwrap()).is_ok());
    let mut bad = p.clone();
    bad.code[target].op = 0xff;
    assert!(bad.bytes().is_err());
}

#[test]
fn atomic_pairs_and_bank_repair() {
    for op in [0x52, 0x55] {
        for imm in 0..10 {
            let i = Inst {
                d: 2,
                a: 4,
                b: 8,
                c: 3,
                imm,
                ..Inst::new(op)
            };
            assert_eq!(Inst::decode(i.encode().unwrap()).unwrap(), i);
            assert_eq!(i.regs(false).contains(&Reg(Class::V, 9)), imm == 2);
        }
    }
    assert!(Inst {
        a: 4,
        b: 9,
        imm: 2,
        ..Inst::new(0x52)
    }
    .validate()
    .is_err());
    assert!(Inst {
        imm: 10,
        ..Inst::new(0x55)
    }
    .validate()
    .is_err());
    let ops = [
        Op::new(0x20, 0, 0, 0, 0, 1),
        Op::new(0x20, 4, 0, 0, 0, 2),
        Op::new(0x20, 8, 0, 0, 0, 3),
        Op::new(0x32, 12, 0, 4, 8, 0),
    ];
    let p = mir::compile(&ops, 0, 0).unwrap();
    assert!(p.code.iter().any(|i| i.op == 0x21));
    assert!(p.code.iter().all(|i| i.bank_legal()));
}
