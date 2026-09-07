// SPDX-License-Identifier: MIT
// Source-neutral SSA values. IDs name typed values, never NIR pointers or opcodes.
use crate::isa::{Class, Inst, Program};
use crate::schedule;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub struct Op {
    pub op: u8,
    pub args: [u32; 4],
    pub imm: u32,
}
impl Op {
    pub fn new(op: u8, d: u32, a: u32, b: u32, c: u32, imm: u32) -> Self {
        Self {
            op,
            args: [d, a, b, c],
            imm,
        }
    }
}
#[derive(Clone, Copy)]
enum Home {
    Register(u8),
    Spill(u32),
}

fn roles(op: u8, imm: u32) -> Result<[Option<(Class, u8)>; 4], String> {
    if op == 0xf0 {
        Ok([Some((Class::V, 2)), Some((Class::V, 1)), None, None])
    } else if op == 0xf1 {
        Ok([
            Some((Class::V, 2)),
            Some((Class::V, 1)),
            Some((Class::V, 1)),
            None,
        ])
    } else if op == 0xf2 && imm < 2 {
        Ok([Some((Class::V, 1)), Some((Class::S, 2)), None, None])
    } else {
        Inst {
            imm,
            ..Inst::new(op)
        }
        .roles()
    }
}

pub fn compile(ops: &[Op], shared: u32, source_private: u32) -> Result<Program, String> {
    if source_private % 4 != 0 {
        return Err("unaligned source private size".into());
    }
    // Out-of-SSA phi webs use mutable virtual registers. Preserve their homes
    // across every mask arm and backedge; linear SSA keeps interval reuse.
    let control = ops.iter().any(|o| matches!(o.op, 4 | 5 | 6));
    let mut values: BTreeMap<u32, (Class, u8, usize, usize)> = BTreeMap::new();
    for (pc, o) in ops.iter().enumerate() {
        if o.op == 3 {
            return Err("MIR scheduler owns waits".into());
        }
        if matches!(o.op, 4 | 5) && o.imm as usize >= ops.len() {
            return Err("MIR branch target outside program".into());
        }
        let roles = roles(o.op, o.imm)?;
        for (field, role) in roles.iter().enumerate() {
            if role.is_none() && o.args[field] != 0 {
                return Err("reserved MIR operand".into());
            }
        }
        if o.op < 0xf0 {
            Inst {
                imm: o.imm,
                ..Inst::new(o.op)
            }
            .validate()?;
        } else if o.op != 0xf2 && o.imm != 0 {
            return Err("reserved MIR immediate".into());
        }
        if let Some((cl, n)) = roles[0] {
            if let Some(v) = values.get_mut(&o.args[0]) {
                if !control || (v.0, v.1) != (cl, n) {
                    return Err("MIR redefinition/type mismatch".into());
                }
                v.3 = pc;
            } else {
                values.insert(o.args[0], (cl, n, pc, pc));
            }
        }
    }
    for (pc, o) in ops.iter().enumerate() {
        for (f, role) in roles(o.op, o.imm)?.iter().enumerate().skip(1) {
            if let Some((cl, n)) = role {
                let v = values.get_mut(&o.args[f]).ok_or("undefined MIR value")?;
                if (v.0, v.1) != (*cl, *n) || (!control && v.2 >= pc) {
                    return Err("MIR type/order mismatch".into());
                }
                v.3 = v.3.max(pc);
            }
        }
    }
    if control {
        for v in values.values_mut() {
            v.2 = 0;
            v.3 = ops.len();
        }
    }
    let mut order: Vec<_> = values.iter().collect();
    order.sort_by_key(|(&id, v)| (v.2, id));
    let mut homes = BTreeMap::new();
    let mut occupied: Vec<(Class, u8, u8, usize)> = Vec::new();
    let mut words = source_private / 4;
    for (&id, &(cl, n, start, end)) in order {
        occupied.retain(|v| v.3 >= start);
        let first = if cl == Class::S { 4 } else { 0 };
        // Rotate preferred bank with SSA identity; reserve 52..63 for reloads and bank repair.
        let free = (first..52)
            .filter(|r| *r + n <= 52 && (n == 1 || r % 2 == 0))
            .filter(|r| {
                !occupied
                    .iter()
                    .any(|v| v.0 == cl && *r < v.1 + v.2 && v.1 < *r + n)
            })
            .min_by_key(|r| ((r % 4 + 4 - (id % 4) as u8) % 4, *r));
        let home = if let Some(r) = free {
            occupied.push((cl, r, n, end));
            Home::Register(r)
        } else {
            if cl == Class::S {
                return Err("scalar register pressure exceeds initial profile".into());
            }
            let slot = words;
            words = words.checked_add(n as u32).ok_or("spill size overflow")?;
            Home::Spill(slot)
        };
        homes.insert(id, home);
    }
    let mut native = Vec::new();
    let mut instruction_map = Vec::new();
    for o in ops {
        instruction_map.push(native.len());
        let roles = roles(o.op, o.imm)?;
        let mut fields = [0; 4];
        let mut stores = Vec::new();
        for f in 0..4 {
            if let Some((_, n)) = roles[f] {
                fields[f] = match homes[&o.args[f]] {
                    Home::Register(r) => r,
                    Home::Spill(slot) => {
                        let r = 56 + f as u8 * 2;
                        for j in 0..n {
                            if f == 0 {
                                stores.push(Inst {
                                    b: r + j,
                                    imm: slot + j as u32,
                                    ..Inst::new(0x57)
                                });
                            } else {
                                native.push(Inst {
                                    d: r + j,
                                    imm: slot + j as u32,
                                    ..Inst::new(0x56)
                                });
                            }
                        }
                        r
                    }
                };
            }
        }
        let mut i = Inst {
            op: o.op,
            d: fields[0],
            a: fields[1],
            b: fields[2],
            c: fields[3],
            imm: o.imm,
        };
        if o.op == 0xf2 {
            native.push(Inst {
                d: i.d,
                a: i.a + o.imm as u8,
                ..Inst::new(0x2d)
            });
            native.extend(stores);
            continue;
        }
        if o.op == 0xf1 {
            native.extend([
                Inst {
                    d: i.d,
                    a: i.a,
                    ..Inst::new(0x21)
                },
                Inst {
                    d: i.d + 1,
                    a: i.b,
                    ..Inst::new(0x21)
                },
            ]);
            native.extend(stores);
            continue;
        }
        if o.op == 0xf0 {
            native.extend([
                Inst {
                    d: i.d,
                    a: i.a,
                    ..Inst::new(0x21)
                },
                Inst {
                    d: i.d + 1,
                    ..Inst::new(0x20)
                },
                Inst {
                    d: 52,
                    a: 0,
                    ..Inst::new(0x2d)
                },
                Inst {
                    d: 53,
                    a: 1,
                    ..Inst::new(0x2d)
                },
                Inst {
                    d: i.d,
                    a: i.d,
                    b: 52,
                    ..Inst::new(0x2c)
                },
            ]);
            native.extend(stores);
            continue;
        }
        if !i.bank_legal() {
            // Only ternary or pair instructions can exceed the two-read bank budget.
            // Copy one operand to a reserved bank/pair with legal read geometry.
            let mut fixed = false;
            for f in 1..4 {
                if let Some((cl, n)) = roles[f] {
                    for r in 52..56 {
                        if n == 2 && (r % 2 != 0 || r + 1 >= 56) {
                            continue;
                        }
                        let mut candidate = fields;
                        candidate[f] = r;
                        let c = Inst {
                            a: candidate[1],
                            b: candidate[2],
                            c: candidate[3],
                            ..i
                        };
                        if c.bank_legal() {
                            for j in 0..n {
                                native.push(Inst {
                                    d: r + j,
                                    a: fields[f] + j,
                                    ..Inst::new(if cl == Class::S { 0x11 } else { 0x21 })
                                });
                            }
                            i = c;
                            fixed = true;
                            break;
                        }
                    }
                    if fixed {
                        break;
                    }
                }
            }
            if !fixed {
                return Err("cannot repair operand banks".into());
            }
        }
        native.push(i);
        native.extend(stores);
    }
    for i in &mut native {
        if matches!(i.op, 4 | 5) {
            i.imm =
                u32::try_from(instruction_map[i.imm as usize]).map_err(|_| "code size overflow")?;
        }
    }
    if native.last().is_none_or(|i| !matches!(i.op, 1 | 2)) {
        native.push(Inst::new(1));
    }
    let code = schedule::schedule(&native)?;
    schedule::validate(&code)?;
    let mut p = Program {
        code,
        entry: 0,
        scalar: 4,
        vector: 0,
        shared,
        private: words.checked_mul(4).ok_or("private bytes overflow")?,
    };
    for i in &p.code {
        for r in i.regs(false).into_iter().chain(i.regs(true)) {
            let count = if r.0 == Class::S {
                &mut p.scalar
            } else {
                &mut p.vector
            };
            *count = (*count).max(r.1 as u32 + 1);
        }
    }
    p.validate()?;
    Ok(p)
}
