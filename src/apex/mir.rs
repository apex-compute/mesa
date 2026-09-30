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
    // A single-definition vector immediate is rebuilt at each use.
    Remat(u32),
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

pub fn compile(
    ops: &[Op],
    shared: u32,
    source_private: u32,
    invocations: u32,
) -> Result<Program, String> {
    if source_private % 4 != 0 {
        return Err("unaligned source private size".into());
    }
    let coalesced;
    let ops = if ops.iter().all(|o| !matches!(o.op, 4 | 5) || (o.imm as usize) < ops.len()) {
        coalesced = coalesce(ops)?;
        &coalesced[..]
    } else {
        ops
    };
    // Masked vector writes preserve inactive lanes in out-of-SSA phi webs.
    // Include every definition/use and enclose backedges before reusing homes.
    let control = ops.iter().any(|o| matches!(o.op, 4 | 5 | 6));
    // Class, width, interval start/end, and the unique definition (if any).
    let mut values: BTreeMap<u32, (Class, u8, usize, usize, Option<usize>)> = BTreeMap::new();
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
                v.4 = None;
            } else {
                values.insert(o.args[0], (cl, n, pc, pc, Some(pc)));
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
                v.2 = v.2.min(pc);
                v.3 = v.3.max(pc);
            }
        }
    }
    if control {
        lifetimes(ops, &mut values)?;
    }
    let mut order: Vec<_> = values.iter().map(|(&id, &v)| (id, v)).collect();
    order.sort_by_key(|&(id, v)| (v.2, id));
    let mut homes = BTreeMap::new();
    // Active register intervals: class, register, width, end, value.
    let mut occupied: Vec<(Class, u8, u8, usize, u32)> = Vec::new();
    // Spill slots: first word, width, start, end. Disjoint lifetimes share slots.
    let mut slots: Vec<(u32, u8, usize, usize)> = Vec::new();
    let reserved = source_private / 4;
    let mut words = reserved;
    let mut spill = |n: u8, start: usize, end: usize, words: &mut u32| -> Result<u32, String> {
        let mut slot = reserved;
        loop {
            let clash = slots
                .iter()
                .find(|s| slot < s.0 + s.1 as u32 && s.0 < slot + n as u32 && start <= s.3 && s.2 <= end);
            match clash {
                Some(s) => slot = s.0 + s.1 as u32,
                None => break,
            }
        }
        slots.push((slot, n, start, end));
        *words = (*words).max(slot.checked_add(n as u32).ok_or("spill size overflow")?);
        Ok(slot)
    };
    let remat = |id: u32| match values[&id] {
        (Class::V, 1, _, _, Some(pc)) if ops[pc].op == 0x20 => Some(ops[pc].imm),
        _ => None,
    };
    // Spill cost: each definition and use weighted by loop depth. A reload
    // costs a load and its wait; a rematerialization one immediate.
    let loops: Vec<(usize, usize)> = ops
        .iter()
        .enumerate()
        .filter(|(pc, o)| matches!(o.op, 4 | 5) && (o.imm as usize) <= *pc)
        .map(|(pc, o)| (o.imm as usize, pc))
        .collect();
    let mut cost: BTreeMap<u32, u64> = BTreeMap::new();
    for (pc, o) in ops.iter().enumerate() {
        let depth = loops.iter().filter(|&&(h, l)| h <= pc && pc <= l).count().min(6);
        for (f, role) in roles(o.op, o.imm)?.iter().enumerate() {
            if role.is_some() {
                *cost.entry(o.args[f]).or_default() += 8u64.pow(depth as u32);
            }
        }
    }
    // Divide by the interval length: spilling a long, rarely used value
    // relieves more pressure than spilling a short, busy one.
    let cost = |id: u32| {
        let (_, _, start, end, _) = values[&id];
        (cost[&id] << 20) * if remat(id).is_some() { 1 } else { 2 } / (end - start + 1) as u64
    };
    for (id, (cl, n, start, end, _)) in order {
        occupied.retain(|v| v.3 >= start);
        let first = if cl == Class::S { 4 } else { 0 };
        let aligned = |r: u8| r + n <= 52 && (n == 1 || r % 2 == 0);
        let overlaps = |r: u8, v: &(Class, u8, u8, usize, u32)| v.0 == cl && r < v.1 + v.2 && v.1 < r + n;
        // Rotate preferred bank with SSA identity; reserve 52..63 for reloads and bank repair.
        let free = (first..52)
            .filter(|&r| aligned(r) && !occupied.iter().any(|v| overlaps(r, v)))
            .min_by_key(|r| ((r % 4 + 4 - (id % 4) as u8) % 4, *r));
        if let Some(r) = free {
            occupied.push((cl, r, n, end, id));
            homes.insert(id, Home::Register(r));
            continue;
        }
        if cl == Class::S {
            return Err("scalar register pressure exceeds initial profile".into());
        }
        // Evict the aligned window whose occupants cost least to spill, then
        // whose nearest end is latest, unless spilling this value costs less.
        // A pair may evict two single values.
        let victim = (first..52)
            .filter(|&r| aligned(r))
            .map(|r| {
                let held = occupied.iter().filter(|v| overlaps(r, v));
                let price: u64 = held.clone().map(|v| cost(v.4)).sum();
                (price, std::cmp::Reverse(held.map(|v| v.3).min().unwrap_or(usize::MAX)), r)
            })
            .min()
            .filter(|&(price, nearest, _)| (price, nearest) < (cost(id), std::cmp::Reverse(end)))
            .map(|(_, _, r)| r);
        let mut home = |id: u32, n: u8, start: usize, end: usize, words: &mut u32| -> Result<Home, String> {
            Ok(match remat(id) {
                Some(imm) => Home::Remat(imm),
                None => Home::Spill(spill(n, start, end, words)?),
            })
        };
        if let Some(r) = victim {
            for (_, _, victim_n, victim_end, victim_id) in occupied.extract_if(.., |v| overlaps(r, v)).collect::<Vec<_>>() {
                let (_, _, victim_start, _, _) = values[&victim_id];
                homes.insert(victim_id, home(victim_id, victim_n, victim_start, victim_end, &mut words)?);
            }
            occupied.push((cl, r, n, end, id));
            homes.insert(id, Home::Register(r));
        } else {
            homes.insert(id, home(id, n, start, end, &mut words)?);
        }
    }
    let mut native = Vec::new();
    let mut instruction_map = Vec::new();
    for o in ops {
        instruction_map.push(native.len());
        let roles = roles(o.op, o.imm)?;
        if roles[0].is_some() && matches!(homes[&o.args[0]], Home::Remat(_)) {
            continue;
        }
        let mut fields = [0; 4];
        let mut stores = Vec::new();
        for f in 0..4 {
            if let Some((_, n)) = roles[f] {
                fields[f] = match homes[&o.args[f]] {
                    Home::Register(r) => r,
                    // Remat definitions were skipped above, so this is a use.
                    Home::Remat(imm) => {
                        let r = 56 + f as u8 * 2;
                        native.push(Inst {
                            d: r,
                            imm,
                            ..Inst::new(0x20)
                        });
                        r
                    }
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
        invocations,
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

/// Removes out-of-SSA vector copies inside one mask region: a copy's single
/// use reads the source register directly, and a single-use value is defined
/// straight into the register it is copied to. Loads then write their final
/// register and overlap instead of waiting at each copy.
fn coalesce(ops: &[Op]) -> Result<Vec<Op>, String> {
    let mut ops = ops.to_vec();
    let roles: Vec<_> = ops.iter().map(|o| roles(o.op, o.imm)).collect::<Result<_, _>>()?;
    let mut boundary = vec![false; ops.len() + 1];
    for (pc, o) in ops.iter().enumerate() {
        if matches!(o.op, 1 | 2 | 4 | 5 | 6 | 7 | 8) {
            boundary[pc] = true;
        }
        if matches!(o.op, 4 | 5) {
            boundary[o.imm as usize] = true;
        }
    }
    let mut defs: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut uses: BTreeMap<u32, Vec<(usize, usize)>> = BTreeMap::new();
    for (pc, (o, r)) in ops.iter().zip(&roles).enumerate() {
        if r[0].is_some() {
            defs.entry(o.args[0]).or_default().push(pc);
        }
        for f in 1..4 {
            if r[f].is_some() {
                uses.entry(o.args[f]).or_default().push((pc, f));
            }
        }
    }
    let single = |defs: &BTreeMap<u32, Vec<usize>>, id: u32| defs.get(&id).map(Vec::len) == Some(1);
    // Whether register id is untouched strictly between pcs a and b in one region.
    let quiet = |ops: &[Op], a: usize, b: usize, id: u32| {
        (a + 1..=b).all(|pc| !boundary[pc])
            && (a + 1..b).all(|pc| (0..4).all(|f| roles[pc][f].is_none() || ops[pc].args[f] != id))
    };
    let vector = |r: Option<(Class, u8)>| r == Some((Class::V, 1));
    let mut removed = vec![false; ops.len()];
    for pc in 0..ops.len() {
        if ops[pc].op != 0x21 {
            continue;
        }
        let (copy, source) = (ops[pc].args[0], ops[pc].args[1]);
        if copy == source {
            continue;
        }
        let copy_uses = uses.get(&copy).cloned().unwrap_or_default();
        // A load_reg copy: forward the register to every use in its region.
        if single(&defs, copy)
            && !copy_uses.is_empty()
            && copy_uses.iter().all(|&(u, _)| u > pc && quiet(&ops, pc, u, source) && !removed[u])
            && copy_uses.iter().all(|&(u, f)| (1..4).all(|g| g == f || ops[u].args[g] != copy))
        {
            for &(u, f) in &copy_uses {
                ops[u].args[f] = source;
            }
            let reads = uses.entry(source).or_default();
            reads.retain(|&(u, _)| u != pc);
            reads.extend(copy_uses);
            removed[pc] = true;
            continue;
        }
        // A store_reg copy: define the single-use source directly into the copy.
        let source_uses = uses.get(&source).map(Vec::as_slice).unwrap_or_default();
        if let (true, [(u, _)], Some(&[d])) = (single(&defs, source), source_uses, defs.get(&source).map(Vec::as_slice)) {
            if *u == pc && d < pc && !removed[d] && vector(roles[d][0]) && quiet(&ops, d, pc, copy)
                && (1..4).all(|f| roles[d][f].is_none() || ops[d].args[f] != source)
            {
                ops[d].args[0] = copy;
                let writes = defs.get_mut(&copy).unwrap();
                writes.retain(|&w| w != pc);
                writes.push(d);
                removed[pc] = true;
            }
        }
    }
    let mut map = vec![0; ops.len() + 1];
    let mut kept = 0;
    for pc in 0..ops.len() {
        map[pc] = kept;
        kept += !removed[pc] as usize;
    }
    map[ops.len()] = kept;
    Ok(ops
        .into_iter()
        .zip(removed)
        .filter(|(_, r)| !r)
        .map(|(mut o, _)| {
            if matches!(o.op, 4 | 5) {
                o.imm = map[o.imm as usize] as u32;
            }
            o
        })
        .collect())
}

/// Hull intervals from liveness on the linear control-flow graph. Every
/// branch edge is taken or skipped as a whole wave, so both sides of a masked
/// region execute in program order. Scalar writes are unmasked and kill. A
/// single vector definition kills only when every use stays inside each loop
/// containing it: lanes that leave a loop early keep an older value in the
/// register. Multi-definition (out-of-SSA) vector values are masked
/// read-modify-writes: they are live from any reaching definition to any use.
fn lifetimes(
    ops: &[Op],
    values: &mut BTreeMap<u32, (Class, u8, usize, usize, Option<usize>)>,
) -> Result<(), String> {
    let n = ops.len();
    let mut leader = vec![false; n + 1];
    leader[0] = true;
    leader[n] = true;
    for (pc, o) in ops.iter().enumerate() {
        if matches!(o.op, 4 | 5) {
            leader[o.imm as usize] = true;
            leader[pc + 1] = true;
        } else if matches!(o.op, 1 | 2) {
            leader[pc + 1] = true;
        }
    }
    let starts: Vec<usize> = (0..=n).filter(|&pc| leader[pc]).collect();
    let blocks = starts.len() - 1;
    let mut block_of = vec![0; n + 1];
    for b in 0..blocks {
        for pc in starts[b]..starts[b + 1] {
            block_of[pc] = b;
        }
    }
    block_of[n] = blocks;
    let mut successors = vec![Vec::new(); blocks];
    for b in 0..blocks {
        let last = &ops[starts[b + 1] - 1];
        let next = (b + 1 < blocks).then_some(b + 1);
        successors[b] = match last.op {
            1 | 2 => vec![],
            4 => vec![block_of[last.imm as usize]],
            5 => [Some(block_of[last.imm as usize]), next].into_iter().flatten().collect(),
            _ => next.into_iter().collect(),
        };
        successors[b].retain(|&s| s < blocks);
    }
    let ids: Vec<u32> = values.keys().copied().collect();
    let index: BTreeMap<u32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let words = ids.len().div_ceil(64);
    let mut defs = vec![Vec::new(); ids.len()];
    let mut uses = vec![Vec::new(); ids.len()];
    for (pc, o) in ops.iter().enumerate() {
        let r = roles(o.op, o.imm)?;
        if r[0].is_some() {
            defs[index[&o.args[0]]].push(pc);
        }
        for f in 1..4 {
            if r[f].is_some() {
                uses[index[&o.args[f]]].push(pc);
            }
        }
    }
    let loops: Vec<(usize, usize)> = ops
        .iter()
        .enumerate()
        .filter(|(pc, o)| matches!(o.op, 4 | 5) && (o.imm as usize) <= *pc)
        .map(|(pc, o)| (o.imm as usize, pc))
        .collect();
    let kills: Vec<bool> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            values[id].0 == Class::S
                || (defs[i].len() == 1
                    && loops.iter().all(|&(h, l)| {
                        !(h <= defs[i][0] && defs[i][0] <= l)
                            || uses[i].iter().all(|&u| h <= u && u <= l)
                    }))
        })
        .collect();
    let set = |bits: &mut Vec<u64>, i: usize| bits[i / 64] |= 1 << (i % 64);
    let clear = |bits: &mut Vec<u64>, i: usize| bits[i / 64] &= !(1 << (i % 64));
    let test = |bits: &Vec<u64>, i: usize| bits[i / 64] >> (i % 64) & 1 != 0;
    let mut gen = vec![vec![0u64; words]; blocks];
    let mut killed = vec![vec![0u64; words]; blocks];
    let mut defined = vec![vec![0u64; words]; blocks];
    for b in 0..blocks {
        for pc in (starts[b]..starts[b + 1]).rev() {
            let o = &ops[pc];
            let r = roles(o.op, o.imm)?;
            if r[0].is_some() {
                let i = index[&o.args[0]];
                set(&mut defined[b], i);
                if kills[i] {
                    clear(&mut gen[b], i);
                    set(&mut killed[b], i);
                }
            }
            for f in 1..4 {
                if r[f].is_some() {
                    set(&mut gen[b], index[&o.args[f]]);
                }
            }
        }
    }
    let mut live_in = vec![vec![0u64; words]; blocks];
    let mut live_out = vec![vec![0u64; words]; blocks];
    loop {
        let mut changed = false;
        for b in (0..blocks).rev() {
            let mut out = vec![0u64; words];
            for &s in &successors[b] {
                for w in 0..words {
                    out[w] |= live_in[s][w];
                }
            }
            let input: Vec<u64> = (0..words).map(|w| gen[b][w] | (out[w] & !killed[b][w])).collect();
            changed |= input != live_in[b] || out != live_out[b];
            live_in[b] = input;
            live_out[b] = out;
        }
        if !changed {
            break;
        }
    }
    let mut predecessors = vec![Vec::new(); blocks];
    for b in 0..blocks {
        for &s in &successors[b] {
            predecessors[s].push(b);
        }
    }
    let mut reach_in = vec![vec![0u64; words]; blocks];
    let mut reach_out = vec![vec![0u64; words]; blocks];
    loop {
        let mut changed = false;
        for b in 0..blocks {
            let mut input = vec![0u64; words];
            for &p in &predecessors[b] {
                for w in 0..words {
                    input[w] |= reach_out[p][w];
                }
            }
            let out: Vec<u64> = (0..words).map(|w| input[w] | defined[b][w]).collect();
            changed |= input != reach_in[b] || out != reach_out[b];
            reach_in[b] = input;
            reach_out[b] = out;
        }
        if !changed {
            break;
        }
    }
    for (i, id) in ids.iter().enumerate() {
        let v = values.get_mut(id).unwrap();
        for &pc in defs[i].iter().chain(&uses[i]) {
            v.2 = v.2.min(pc);
            v.3 = v.3.max(pc);
        }
        for b in 0..blocks {
            if test(&live_in[b], i) && test(&reach_in[b], i) {
                v.2 = v.2.min(starts[b]);
                v.3 = v.3.max(starts[b]);
            }
            if test(&live_out[b], i) && test(&reach_out[b], i) {
                v.3 = v.3.max(starts[b + 1] - 1);
            }
        }
    }
    Ok(())
}
