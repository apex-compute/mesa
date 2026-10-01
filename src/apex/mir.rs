// SPDX-License-Identifier: MIT
// Machine IR: ISA opcodes over typed scalar and vector values. Fields map
// one-to-one onto the instruction's d, a, b and c; the value table gives each
// value its class and width in dwords.
use crate::isa::{self, op, Access, Class, Format, Inst, Kind, Program, EXEC, LITERAL, SCALAR};
use crate::schedule;
use std::collections::BTreeMap;

/// f0 <- imm (one dword).
pub const CONST: u16 = 0x100;
/// f0 <- f1, dword by dword; a vector source into a scalar reads the first active lane.
pub const COPY: u16 = 0x101;
/// Branch target `imm`.
pub const LABEL: u16 = 0x102;
/// f0 is the launch register `hi` (defined before the first instruction).
pub const ENTRY: u16 = 0x103;
/// Memory access that no store of this program can alias.
pub const REORDER: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opnd {
    None,
    Val { id: u32, off: u8, n: u8 },
    Phys { code: u8, n: u8 },
    Raw(u8),
    Lit(u32),
}
impl Opnd {
    pub fn val(self) -> Option<u32> {
        match self {
            Opnd::Val { id, .. } => Some(id),
            _ => None,
        }
    }
    pub fn n(self) -> u8 {
        match self {
            Opnd::Val { n, .. } | Opnd::Phys { n, .. } => n,
            _ => 1,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Op {
    pub op: u16,
    pub f: [Opnd; 4],
    pub hi: u32,
    pub imm: u32,
    pub flags: u32,
}
impl Op {
    pub fn new(op: u16, f: [Opnd; 4]) -> Self {
        Self { op, f, hi: 0, imm: 0, flags: 0 }
    }
    pub fn isa(&self) -> Option<u8> {
        (self.op < 0x100).then_some(self.op as u8)
    }
    pub fn access(&self) -> [Option<Access>; 4] {
        match self.op {
            CONST | ENTRY => [Some(Access::Def), None, None, None],
            COPY => [Some(Access::Def), Some(Access::Use), None, None],
            LABEL => [None; 4],
            o => {
                let kinds = isa::fields(o as u8, self.hi).unwrap_or([Kind::None; 4]);
                std::array::from_fn(|f| match kinds[f] {
                    Kind::S(a, _) | Kind::V(a, _) => Some(a),
                    Kind::Any | Kind::SAny | Kind::OptS(_) | Kind::OptV(_) => Some(Access::Use),
                    Kind::None | Kind::Raw => None,
                })
            }
        }
    }
    pub fn defs(&self) -> impl Iterator<Item = u32> + '_ {
        let a = self.access();
        (0..4).filter(move |&f| matches!(a[f], Some(Access::Def | Access::DefUse))).filter_map(|f| self.f[f].val())
    }
    pub fn uses(&self) -> impl Iterator<Item = u32> + '_ {
        let a = self.access();
        (0..4).filter(move |&f| matches!(a[f], Some(Access::Use | Access::DefUse))).filter_map(|f| self.f[f].val())
    }
    pub fn writes_exec(&self) -> bool {
        matches!(self.f[0], Opnd::Phys { code: EXEC, .. }) && self.op != LABEL
            || matches!(self.isa(), Some(op::S_AND_SAVEEXEC..=op::S_SETEXEC))
    }
    pub fn branch(&self) -> bool {
        matches!(self.isa(), Some(op::S_BRANCH..=op::S_CBRANCH_EXECNZ))
    }
    pub fn terminal(&self) -> bool {
        matches!(self.isa(), Some(op::S_ENDPGM | op::S_TRAP))
    }
    /// Ordered with every other memory or export operation.
    pub fn side_effect(&self) -> bool {
        match self.isa() {
            Some(o) => matches!(o, op::GLOBAL_STORE | op::GLOBAL_ATOMIC | op::BUFFER_STORE | op::BUFFER_ATOMIC
                | op::SCRATCH_STORE | op::SHARED_STORE | op::SHARED_ATOMIC | op::EXP)
                || isa::format(o) == Some(Format::Control),
            None => self.op == LABEL,
        }
    }
    pub fn load(&self) -> bool {
        matches!(self.isa(), Some(op::S_LOAD..=op::GLOBAL_LOAD | op::BUFFER_LOAD | op::SCRATCH_LOAD
            | op::SHARED_LOAD | op::IMAGE_SAMPLE | op::IMAGE_FETCH))
    }
    /// Instructions that bound scheduling regions.
    pub fn boundary(&self) -> bool {
        self.op == LABEL || self.op == ENTRY || self.writes_exec()
            || matches!(self.isa().and_then(isa::format), Some(Format::Control))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Value {
    pub class: Class,
    pub width: u8,
}
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub instructions: u32,
    pub vector: u32,
    pub scalar: u32,
    pub spills: u32,
}

fn check(ops: &[Op], values: &[Value]) -> Result<(), String> {
    for (pc, o) in ops.iter().enumerate() {
        let fail = |m: &str| Err(format!("MIR op {pc} (0x{:x}): {m}", o.op));
        let class_of = |x: Opnd| match x {
            Opnd::Val { id, off, n } => {
                let v = values.get(id as usize).filter(|_| id != 0)?;
                (off + n <= v.width).then_some(v.class)
            }
            Opnd::Phys { code, .. } => Some(if code < SCALAR { Class::V } else { Class::S }),
            _ => None,
        };
        for (f, x) in o.f.iter().enumerate() {
            if let Opnd::Val { .. } = x {
                if class_of(*x).is_none() {
                    return fail(&format!("field {f} names an undeclared value or range"));
                }
            }
        }
        let kinds = match o.op {
            CONST | ENTRY => {
                if o.f[0].val().is_none() || o.f[0].n() != 1 {
                    return fail("needs one destination dword");
                }
                continue;
            }
            COPY => {
                if o.f[0].n() != o.f[1].n() || class_of(o.f[0]).is_none()
                    || !matches!(o.f[1], Opnd::Val { .. } | Opnd::Phys { .. } | Opnd::Lit(_))
                {
                    return fail("copy widths");
                }
                continue;
            }
            LABEL => continue,
            x if x < 0x100 => isa::fields(x as u8, o.hi)?,
            _ => return fail("unknown opcode"),
        };
        for f in 0..4 {
            let x = o.f[f];
            let class = class_of(x);
            let ok = match kinds[f] {
                Kind::None => x == Opnd::None,
                Kind::Raw => matches!(x, Opnd::Raw(_)),
                Kind::S(Access::Def, 1) if f == 0 => class == Some(Class::S) && x.n() == 1,
                Kind::S(_, n) => class == Some(Class::S) && x.n() == n,
                Kind::V(_, n) => class == Some(Class::V) && x.n() == n,
                Kind::Any => x.n() == 1 && !matches!(x, Opnd::None | Opnd::Raw(_)),
                Kind::SAny => x.n() == 1 && matches!(class, Some(Class::S) | None) && !matches!(x, Opnd::None | Opnd::Raw(_)),
                Kind::OptS(n) => x == Opnd::None || (class == Some(Class::S) && x.n() == n),
                Kind::OptV(n) => {
                    let n = if matches!(o.op as u8, op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC) && o.f[2] != Opnd::None { 1 } else { n };
                    x == Opnd::None || (class == Some(Class::V) && x.n() == n)
                }
            };
            if !ok {
                return fail(&format!("field {f} does not match {:?}", kinds[f]));
            }
        }
    }
    Ok(())
}

/// Replaces single-dword constants by inline codes or the instruction's literal
/// where the encoding allows, then materializes the rest.
fn fold_constants(ops: &mut [Op], values: &[Value]) {
    // Dwords written once, by a constant: (value, dword) -> (constant, writes).
    let mut constant: BTreeMap<(u32, u8), (u32, usize)> = BTreeMap::new();
    for o in ops.iter() {
        let access = o.access();
        for f in 0..4 {
            if !matches!(access[f], Some(Access::Def | Access::DefUse)) {
                continue;
            }
            let Opnd::Val { id, off, n } = o.f[f] else { continue };
            for k in 0..n {
                let (v, w) = if o.op == CONST { (o.imm, 1) } else { (0, 2) };
                constant.entry((id, off + k)).and_modify(|e| e.1 += w).or_insert((v, w));
            }
        }
    }
    constant.retain(|_, e| e.1 == 1);
    for o in ops.iter_mut() {
        let Some(code) = o.isa() else {
            if o.op == COPY {
                if let Opnd::Val { id, off, n: 1 } = o.f[1] {
                    if let Some(&(v, _)) = constant.get(&(id, off)) {
                        o.f[1] = Opnd::Lit(v);
                    }
                }
            }
            continue;
        };
        let kinds = isa::fields(code, o.hi).unwrap();
        let alu = matches!(isa::format(code), Some(Format::Salu | Format::Valu));
        let mut literal: Option<u32> = None;
        for f in 1..4 {
            if !matches!(kinds[f], Kind::Any | Kind::SAny) {
                continue;
            }
            let Opnd::Val { id, off, n: 1 } = o.f[f] else { continue };
            let Some(&(v, _)) = constant.get(&(id, off)) else { continue };
            if isa::inline_code(v).is_some() {
                o.f[f] = Opnd::Lit(v);
            } else if alu && f < 3 && kinds[3] == Kind::None && (o.hi >> 8) & 0x7f == 0
                && literal.is_none_or(|l| l == v)
            {
                literal = Some(v);
                o.f[f] = Opnd::Lit(v);
            }
        }
    }
    // A constant left in a register shares an equal earlier one of its
    // straight-line region when every read is in that region.
    let mut region = vec![0u32; ops.len()];
    let mut r = 0;
    for (i, o) in ops.iter().enumerate() {
        r += o.boundary() as u32;
        region[i] = r;
    }
    let mut reads: BTreeMap<u32, Option<u32>> = BTreeMap::new();
    for (i, o) in ops.iter().enumerate() {
        for u in o.uses() {
            reads.entry(u).and_modify(|e| *e = e.filter(|&x| x == region[i])).or_insert(Some(region[i]));
        }
    }
    let mut first: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut rename: BTreeMap<u32, u32> = BTreeMap::new();
    for (i, o) in ops.iter().enumerate() {
        let Opnd::Val { id, off: 0, n: 1 } = o.f[0] else { continue };
        if o.op != CONST || values[id as usize].width != 1 || !constant.contains_key(&(id, 0))
            || reads.get(&id).is_some_and(|&x| x != Some(region[i]))
        {
            continue;
        }
        match first.entry((region[i], o.imm)) {
            std::collections::btree_map::Entry::Occupied(e) => {
                rename.insert(id, *e.get());
            }
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(id);
            }
        }
    }
    for o in ops.iter_mut() {
        let access = o.access();
        for f in 1..4 {
            if let (Some(Access::Use), Opnd::Val { id, off, n }) = (access[f], o.f[f]) {
                if let Some(&to) = rename.get(&id) {
                    o.f[f] = Opnd::Val { id: to, off, n };
                }
            }
        }
        if o.op == CONST {
            o.op = COPY;
            o.f[1] = Opnd::Lit(o.imm);
        }
    }
}

/// Mark and sweep: an op is needed for its effect or for a needed value.
fn eliminate_dead(ops: &mut Vec<Op>) {
    let root = |o: &Op| o.side_effect() || o.writes_exec() || (o.defs().next().is_none() && o.op != COPY);
    let mut needed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut keep: Vec<bool> = ops.iter().map(root).collect();
    loop {
        let mut changed = false;
        for (i, o) in ops.iter().enumerate() {
            if !keep[i] && o.defs().any(|d| needed.contains(&d)) {
                keep[i] = true;
                changed = true;
            }
            if keep[i] {
                for u in o.uses() {
                    changed |= needed.insert(u);
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut i = 0;
    ops.retain(|_| {
        i += 1;
        keep[i - 1]
    });
}

/// Launch registers become values with fixed homes defined before the program.
fn bind_launch(ops: &mut Vec<Op>, values: &mut Vec<Value>) -> BTreeMap<u32, u8> {
    let mut fixed = BTreeMap::new();
    let mut by_code: BTreeMap<u8, u32> = BTreeMap::new();
    let mut entries = Vec::new();
    for o in ops.iter_mut() {
        let access = o.access();
        for f in 0..4 {
            let Opnd::Phys { code, n } = o.f[f] else { continue };
            if code >= isa::EXEC || matches!(access[f], Some(Access::Def | Access::DefUse)) {
                continue;
            }
            // Groups resolve to their first register's value; wider views allocate a group.
            let id = *by_code.entry(code).or_insert_with(|| {
                values.push(Value { class: if code < SCALAR { Class::V } else { Class::S }, width: n });
                let id = values.len() as u32 - 1;
                fixed.insert(id, if code < SCALAR { code } else { code - SCALAR });
                entries.push(Op { hi: code as u32, ..Op::new(ENTRY, [Opnd::Val { id, off: 0, n }, Opnd::None, Opnd::None, Opnd::None]) });
                id
            });
            if values[id as usize].width < n {
                values[id as usize].width = n;
                for e in entries.iter_mut().filter(|e| e.f[0].val() == Some(id)) {
                    e.f[0] = Opnd::Val { id, off: 0, n };
                }
            }
            o.f[f] = Opnd::Val { id, off: 0, n };
        }
    }
    // Overlapping launch groups would need one value; the C side reads each register once.
    ops.splice(0..0, entries);
    fixed
}

/// Merges copy sources and targets that can share registers: a value copied
/// into part of a group becomes that part, and a copy of a value becomes the
/// value. The target region must be written only by copies from the source at
/// the same relative offset, and the source must be written once per dword,
/// so no read observes a different write after merging.
fn coalesce(ops: &mut Vec<Op>, values: &[Value], fixed: &mut BTreeMap<u32, u8>) -> Vec<u8> {
    let n = values.len();
    let mut align: Vec<u8> = values.iter().map(alignment).collect();
    let mut parent: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, 0)).collect();
    fn find(parent: &mut Vec<(u32, u32)>, id: u32) -> (u32, u32) {
        let (p, o) = parent[id as usize];
        if p == id {
            return (id, 0);
        }
        let (r, ro) = find(parent, p);
        parent[id as usize] = (r, o + ro);
        (r, o + ro)
    }
    let width = |id: u32| values[id as usize].width as u32;
    for c in 0..ops.len() {
        if ops[c].op != COPY {
            continue;
        }
        let (Opnd::Val { id: d, off: k, .. }, Opnd::Val { id: s, off: j, .. }) = (ops[c].f[0], ops[c].f[1]) else {
            continue;
        };
        if values[d as usize].class != values[s as usize].class {
            continue;
        }
        let (rd, od) = find(&mut parent, d);
        let (rs, os) = find(&mut parent, s);
        if rd == rs {
            continue;
        }
        // Source dword x sits at target dword x + delta.
        let delta = (od + k as u32) as i64 - (os + j as u32) as i64;
        let (small, big, at) = if width(rs) as i64 + delta <= width(rd) as i64 && delta >= 0 {
            (rs, rd, delta)
        } else if width(rd) as i64 - delta <= width(rs) as i64 && delta <= 0 {
            (rd, rs, -delta)
        } else {
            continue;
        };
        let at = at as u32;
        if at % align[small as usize] as u32 != 0 {
            continue;
        }
        let new_fixed = match (fixed.get(&small).copied(), fixed.get(&big).copied()) {
            (Some(fs), Some(fb)) if fb as u32 + at != fs as u32 => continue,
            (Some(fs), None) => {
                if (fs as u32) < at || (fs as u32 - at) % align[big as usize] as u32 != 0 {
                    continue;
                }
                let home = fs as u32 - at;
                let class = values[big as usize].class;
                // Other launch values inside the new home must be copied
                // whole into the matching place of the group.
                let copied = |id: u32, h: u32, parent: &mut Vec<(u32, u32)>| {
                    ops.iter().any(|o| match (o.op, o.f[0], o.f[1]) {
                        (COPY, Opnd::Val { id: a, off: ao, .. }, Opnd::Val { id: s, off: 0, .. }) if s == id => {
                            let (ra, oa) = find(parent, a);
                            ra == big && home + oa + ao as u32 == h
                        }
                        _ => false,
                    })
                };
                let others: Vec<(u32, u8)> = fixed.iter().map(|(&id, &h)| (id, h)).collect();
                let clash = others.into_iter().any(|(id, h)| {
                    id != small && values[id as usize].class == class
                        && (h as u32) < home + width(big) && home < h as u32 + values[id as usize].width as u32
                        && !copied(id, h as u32, &mut parent)
                });
                if clash {
                    continue;
                }
                Some(home as u8)
            }
            (_, fb) => fb,
        };
        // Region [at, at + width(small)) of big, seen through current roots.
        let (target, source) = (rd, rs);
        let coord = |r: u32, x: u32| if r == big { x } else { x + at };
        let mut writes = vec![0u32; width(big) as usize];
        let mut ok = true;
        for o in ops.iter() {
            let access = o.access();
            if o.op == COPY {
                if let (Opnd::Val { id: a, off: ao, .. }, Opnd::Val { id: b, off: bo, .. }) = (o.f[0], o.f[1]) {
                    let (ra, oa) = find(&mut parent, a);
                    let (rb, ob) = find(&mut parent, b);
                    if ra == rb && oa + ao as u32 == ob + bo as u32 {
                        continue;
                    }
                }
            }
            for f in 0..4 {
                if !matches!(access[f], Some(Access::Def | Access::DefUse)) {
                    continue;
                }
                let Opnd::Val { id, off, n } = o.f[f] else { continue };
                let (r, ro) = find(&mut parent, id);
                if r != target && r != source {
                    continue;
                }
                for x in 0..n as u32 {
                    let b = coord(r, ro + off as u32 + x);
                    if b < at || b >= at + width(small) {
                        continue;
                    }
                    if r == target {
                        // Target writes in the region must be copies from the source.
                        let copy = o.op == COPY && f == 0 && match o.f[1] {
                            Opnd::Val { id: sid, off: soff, .. } => {
                                let (sr, so) = find(&mut parent, sid);
                                sr == source && coord(source, so + soff as u32 + x) == b
                            }
                            _ => false,
                        };
                        ok &= copy;
                    } else {
                        writes[b as usize] += 1;
                    }
                }
            }
        }
        if !ok || writes.iter().any(|&w| w > 1) {
            continue;
        }
        parent[small as usize] = (big, at);
        align[big as usize] = align[big as usize].max(align[small as usize]);
        if let Some(h) = new_fixed {
            fixed.insert(big, h);
        }
        fixed.remove(&small);
    }
    for o in ops.iter_mut() {
        for f in 0..4 {
            if let Opnd::Val { id, off, n } = o.f[f] {
                let (r, ro) = find(&mut parent, id);
                o.f[f] = Opnd::Val { id: r, off: (ro + off as u32) as u8, n };
            }
        }
    }
    ops.retain(|o| !(o.op == COPY && o.f[0] == o.f[1]));
    align
}

struct Interval {
    start: usize,
    end: usize,
}

/// Hull intervals from liveness on the linear control-flow graph. Every
/// branch edge is taken or skipped by the whole wave, so both sides of a
/// masked region execute in program order. Scalar writes are unmasked and
/// kill. A single vector definition kills only when every use stays inside
/// each loop containing it: lanes that leave a loop early keep an older
/// value. Multi-definition vector values are masked read-modify-writes and
/// stay live from any reaching definition to any use.
fn lifetimes(ops: &[Op], values: &[Value]) -> Result<BTreeMap<u32, Interval>, String> {
    let n = ops.len();
    let mut labels = BTreeMap::new();
    for (pc, o) in ops.iter().enumerate() {
        if o.op == LABEL {
            labels.insert(o.imm, pc);
        }
    }
    let target = |o: &Op| -> Result<usize, String> {
        labels.get(&o.imm).copied().ok_or_else(|| format!("undefined label {}", o.imm))
    };
    let mut leader = vec![false; n + 1];
    leader[0] = true;
    leader[n] = true;
    for (pc, o) in ops.iter().enumerate() {
        if o.op == LABEL {
            leader[pc] = true;
        }
        if o.branch() || o.terminal() {
            leader[pc + 1] = true;
        }
    }
    let starts: Vec<usize> = (0..=n).filter(|&pc| leader[pc]).collect();
    let blocks = starts.len() - 1;
    let mut block_of = vec![blocks; n + 1];
    for b in 0..blocks {
        for pc in starts[b]..starts[b + 1] {
            block_of[pc] = b;
        }
    }
    let mut successors = vec![Vec::new(); blocks];
    for b in 0..blocks {
        let last = &ops[starts[b + 1] - 1];
        let next = (b + 1 < blocks).then_some(b + 1);
        successors[b] = if last.terminal() {
            vec![]
        } else if last.isa() == Some(op::S_BRANCH) {
            vec![block_of[target(last)?]]
        } else if last.branch() {
            [Some(block_of[target(last)?]), next].into_iter().flatten().collect()
        } else {
            next.into_iter().collect()
        };
        successors[b].retain(|&s| s < blocks);
    }
    let ids: Vec<u32> = {
        let mut s = std::collections::BTreeSet::new();
        for o in ops {
            s.extend(o.defs());
            s.extend(o.uses());
        }
        s.into_iter().collect()
    };
    let index: BTreeMap<u32, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let words = ids.len().div_ceil(64).max(1);
    let mut defs = vec![Vec::new(); ids.len()];
    let mut uses = vec![Vec::new(); ids.len()];
    for (pc, o) in ops.iter().enumerate() {
        for d in o.defs() {
            defs[index[&d]].push(pc);
        }
        for u in o.uses() {
            uses[index[&u]].push(pc);
        }
    }
    let mut loops = Vec::new();
    for (pc, o) in ops.iter().enumerate() {
        if o.branch() {
            let t = target(o)?;
            if t <= pc {
                loops.push((t, pc));
            }
        }
    }
    let kills: Vec<bool> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            values[*id as usize].class == Class::S
                || (defs[i].len() == 1
                    && loops.iter().all(|&(h, l)| {
                        !(h <= defs[i][0] && defs[i][0] <= l) || uses[i].iter().all(|&u| h <= u && u <= l)
                    }))
        })
        .collect();
    // A value accessed only inside one block whose first access writes it
    // (spill temporaries, groups assembled for one instruction) never carries
    // lanes or dwords across blocks: its first write ends the previous contents.
    let mut first: Vec<Option<(usize, usize, bool)>> = vec![None; ids.len()];
    let mut local = vec![true; ids.len()];
    for (pc, o) in ops.iter().enumerate() {
        let a = o.access();
        for f in 0..4 {
            let (Some(access), Opnd::Val { id, .. }) = (a[f], o.f[f]) else { continue };
            let i = index[&id];
            match first[i] {
                None => first[i] = Some((pc, block_of[pc], access == Access::Def && !o.uses().any(|u| u == id))),
                Some((_, b, _)) => local[i] &= b == block_of[pc],
            }
        }
    }
    let set = |bits: &mut Vec<u64>, i: usize| bits[i / 64] |= 1 << (i % 64);
    let clear = |bits: &mut Vec<u64>, i: usize| bits[i / 64] &= !(1 << (i % 64));
    let test = |bits: &Vec<u64>, i: usize| bits[i / 64] >> (i % 64) & 1 != 0;
    let mut gen = vec![vec![0u64; words]; blocks];
    let mut killed = vec![vec![0u64; words]; blocks];
    let mut defined = vec![vec![0u64; words]; blocks];
    for b in 0..blocks {
        for pc in (starts[b]..starts[b + 1]).rev() {
            let o = &ops[pc];
            let a = o.access();
            for f in 0..4 {
                let (Some(access @ (Access::Def | Access::DefUse)), Opnd::Val { id, n, .. }) = (a[f], o.f[f]) else { continue };
                let i = index[&id];
                set(&mut defined[b], i);
                let full = access == Access::Def && n == values[id as usize].width && !o.uses().any(|u| u == id);
                let local_start = local[i] && first[i].is_some_and(|(p, _, write)| p == pc && write);
                if (kills[i] && full) || local_start {
                    clear(&mut gen[b], i);
                    set(&mut killed[b], i);
                }
            }
            for u in o.uses() {
                set(&mut gen[b], index[&u]);
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
    let mut result = BTreeMap::new();
    for (i, id) in ids.iter().enumerate() {
        let mut v = Interval { start: usize::MAX, end: 0 };
        for &pc in defs[i].iter().chain(&uses[i]) {
            v.start = v.start.min(pc);
            v.end = v.end.max(pc);
        }
        for b in 0..blocks {
            if test(&live_in[b], i) && test(&reach_in[b], i) {
                v.start = v.start.min(starts[b]);
                v.end = v.end.max(starts[b]);
            }
            if test(&live_out[b], i) && test(&reach_out[b], i) {
                v.end = v.end.max(starts[b + 1] - 1);
            }
        }
        result.insert(*id, v);
    }
    Ok(result)
}

fn alignment(v: &Value) -> u8 {
    match (v.class, v.width) {
        (_, 1) => 1,
        (Class::V, 2) => 2,
        (Class::V, _) => 1,
        (Class::S, 2 | 3) => 2,
        (Class::S, _) => 4,
    }
}

/// Allocation failure: a message and, when spilling, the vector values chosen
/// to live in private memory.
struct Pressure {
    message: String,
    live: Vec<u32>,
}
impl From<String> for Pressure {
    fn from(message: String) -> Self {
        Self { message, live: Vec::new() }
    }
}
impl From<&str> for Pressure {
    fn from(message: &str) -> Self {
        message.to_string().into()
    }
}

/// Linear scan over lifetime hulls. With `spill`, a vector value that finds
/// no register evicts the live value whose interval ends last (never a spill
/// temporary in `spill`), and the scan continues to collect every victim.
fn allocate(ops: &[Op], values: &[Value], fixed: &BTreeMap<u32, u8>, align: &[u8],
            spill: Option<&std::collections::BTreeSet<u32>>) -> Result<(BTreeMap<u32, u8>, Stats), Pressure> {
    let intervals = lifetimes(ops, values)?;
    // Coalescing hints: a copy prefers its source's registers, and values
    // copied into one group prefer consecutive registers.
    let mut hints: BTreeMap<u32, Vec<(u32, i32)>> = BTreeMap::new();
    let mut members: BTreeMap<u32, Vec<(u32, i32)>> = BTreeMap::new();
    // Copies into a group: group -> (source, offset of the source's dword 0, pc).
    let mut copies: BTreeMap<u32, Vec<(u32, i32, usize)>> = BTreeMap::new();
    for (pc, o) in ops.iter().enumerate() {
        if o.op != COPY {
            continue;
        }
        if let (Opnd::Val { id: d, off: od, .. }, Opnd::Val { id: s, off: os, .. }) = (o.f[0], o.f[1]) {
            copies.entry(d).or_default().push((s, od as i32 - os as i32, pc));
        }
        if let (Opnd::Val { id: d, off: od, .. }, Opnd::Val { id: s, off: os, .. }) = (o.f[0], o.f[1]) {
            if values[d as usize].class == values[s as usize].class && d != s {
                hints.entry(d).or_default().push((s, os as i32 - od as i32));
                hints.entry(s).or_default().push((d, od as i32 - os as i32));
                members.entry(d).or_default().push((s, od as i32 - os as i32));
            }
        }
    }
    // Member x of group g sits at g + dx: home(x) = home(y) + dx - dy.
    let mut group_of: BTreeMap<u32, (u32, i32)> = BTreeMap::new();
    for (&g, list) in &members {
        for &(x, dx) in list {
            group_of.entry(x).or_insert((g, dx));
            for &(y, dy) in list {
                if x != y {
                    hints.entry(x).or_default().push((y, dx - dy));
                }
            }
        }
    }
    let mut order: Vec<u32> = intervals.keys().copied().collect();
    order.sort_by_key(|id| (intervals[id].start, !fixed.contains_key(id), *id));
    let mut homes: BTreeMap<u32, u8> = BTreeMap::new();
    let mut active: Vec<u32> = Vec::new();
    let mut victims: Vec<u32> = Vec::new();
    let mut stats = Stats::default();
    for id in order {
        let v = values[id as usize];
        let iv = &intervals[&id];
        active.retain(|a| intervals[a].end > iv.start);
        let limit = if v.class == Class::S { isa::SCALAR_REGISTERS } else { isa::VECTOR_REGISTERS };
        // A value that dies copying itself into this group at its own place does not conflict.
        let into = copies.get(&id);
        let fits = |r: u8, homes: &BTreeMap<u32, u8>, active: &[u32]| {
            r as u32 + v.width as u32 <= limit as u32
                && r % align[id as usize] == 0
                && !active.iter().any(|a| {
                    let w = values[*a as usize];
                    let h = homes[a];
                    w.class == v.class && r < h + w.width && h < r + v.width
                        && !into.is_some_and(|list| list.iter().any(|&(s, delta, pc)| {
                            s == *a && h as i32 == r as i32 + delta && intervals[a].end == pc
                        }))
                })
        };
        let home = if let Some(&r) = fixed.get(&id) {
            if !fits(r, &homes, &active) {
                return Err("launch register conflict".into());
            }
            r
        } else {
            let hinted = hints.get(&id).into_iter().flatten().find_map(|&(other, delta)| {
                let h = *homes.get(&other)? as i32 + delta;
                ((0..=255).contains(&h) && fits(h as u8, &homes, &active)).then_some(h as u8)
            });
            // The first member of a group to allocate leaves room for the group.
            let grouped = |active: &[u32]| {
                let &(g, dx) = group_of.get(&id)?;
                let gw = values[g as usize].width as u32;
                let ga = align[g as usize] as u32;
                (0..limit as u32).filter(|b| b % ga == 0 && b + gw <= limit as u32).find_map(|b| {
                    let free = (0..gw).all(|k| !active.iter().any(|a| {
                        let w = values[*a as usize];
                        let h = homes[a] as u32;
                        w.class == v.class && b + k >= h && b + k < h + w.width as u32
                    }));
                    let r = b as i32 + dx;
                    (free && r >= 0 && fits(r as u8, &homes, &active)).then_some(r as u8)
                })
            };
            match hinted.or_else(|| grouped(&active)).or_else(|| (0..limit).find(|&r| fits(r, &homes, &active))) {
                Some(r) => r,
                None => {
                    let message = format!("{} register pressure exceeds {} registers",
                        if v.class == Class::S { "scalar" } else { "vector" }, limit);
                    let Some(temps) = spill else {
                        return Err(Pressure { message, live: Vec::new() });
                    };
                    let spillable = |a: u32| !fixed.contains_key(&a) && !temps.contains(&a)
                        && values[a as usize].class == v.class;
                    // Evict the latest-ending live values until this one fits.
                    let mut placed = None;
                    loop {
                        // A value that cannot spill itself (a spill temporary) evicts any.
                        let victim = active.iter().copied().filter(|&a| spillable(a))
                            .max_by_key(|a| intervals[a].end)
                            .filter(|a| intervals[a].end > iv.end || !spillable(id));
                        let Some(victim) = victim else { break };
                        active.retain(|&a| a != victim);
                        victims.push(victim);
                        if let Some(r) = (0..limit).find(|&r| fits(r, &homes, &active)) {
                            placed = Some(r);
                            break;
                        }
                    }
                    match placed {
                        Some(r) => r,
                        None if spillable(id) => {
                            victims.push(id);
                            continue;
                        }
                        None => {
                            return Err(Pressure {
                                message: format!("{message} (value {id}, width {}, {} live)", v.width, active.len()),
                                live: Vec::new(),
                            })
                        }
                    }
                }
            }
        };
        homes.insert(id, home);
        active.push(id);
        let count = if v.class == Class::S { &mut stats.scalar } else { &mut stats.vector };
        *count = (*count).max(home as u32 + v.width as u32);
    }
    if !victims.is_empty() {
        return Err(Pressure { message: "register pressure exceeds the register file".into(), live: victims });
    }
    Ok((homes, stats))
}

fn code(x: Opnd, values: &[Value], homes: &BTreeMap<u32, u8>) -> u8 {
    match x {
        Opnd::None => 0,
        Opnd::Val { id, off, .. } => {
            let r = homes[&id] + off;
            if values[id as usize].class == Class::S { SCALAR + r } else { r }
        }
        Opnd::Phys { code, .. } => code,
        Opnd::Raw(r) => r,
        Opnd::Lit(v) => isa::inline_code(v).unwrap_or(LITERAL),
    }
}

fn lower(ops: &[Op], values: &[Value], homes: &BTreeMap<u32, u8>) -> Result<Vec<Inst>, String> {
    let mut out: Vec<Inst> = Vec::new();
    let mut labels = BTreeMap::new();
    let mut branches = Vec::new();
    for o in ops {
        match o.op {
            LABEL => {
                labels.insert(o.imm, out.len());
            }
            ENTRY => {}
            COPY => {
                let n = o.f[0].n();
                let dst = code(o.f[0], values, homes);
                let src = code(o.f[1], values, homes);
                let lit = if let Opnd::Lit(v) = o.f[1] { v } else { 0 };
                let scalar = dst >= SCALAR;
                let from_vector = src < SCALAR && !matches!(o.f[1], Opnd::Lit(_));
                // Overlapping groups copy away from the overlap.
                let order: Vec<u8> = if dst > src && !matches!(o.f[1], Opnd::Lit(_)) { (0..n).rev().collect() } else { (0..n).collect() };
                for k in order {
                    let (d, a) = (dst + k, if matches!(o.f[1], Opnd::Lit(_)) { src } else { src + k });
                    if d == a {
                        continue;
                    }
                    let mut i = Inst::new(if !scalar {
                        op::V_MOV
                    } else if from_vector {
                        op::V_READFIRSTLANE
                    } else if d == EXEC {
                        op::S_SETEXEC
                    } else {
                        op::S_MOV
                    });
                    i.d = if d == EXEC { 0 } else { d };
                    i.a = a;
                    if a == LITERAL {
                        i.hi = lit;
                    }
                    out.push(i);
                }
            }
            x => {
                let x = x as u8;
                let fmt = isa::format(x).unwrap();
                let kinds = isa::fields(x, o.hi)?;
                let mut i = Inst { op: x, d: 0, a: 0, b: 0, hi: o.hi };
                let mut literal = None;
                for f in 0..4 {
                    let mut c = code(o.f[f], values, homes);
                    if o.f[f] == Opnd::None && matches!(kinds[f], Kind::OptS(_) | Kind::OptV(_)) {
                        c = LITERAL;
                    }
                    if let Opnd::Lit(v) = o.f[f] {
                        if c == LITERAL {
                            literal = Some(v);
                        }
                    }
                    if f < 3 || matches!(fmt, Format::Salu | Format::Valu | Format::Texture) {
                        i.set_field(f, c);
                    }
                }
                if x == op::V_QUADPERM {
                    if let Opnd::Lit(v) = o.f[2] {
                        i.b = LITERAL;
                        literal = Some(v);
                    }
                }
                if let Some(v) = literal {
                    i.hi = v;
                }
                if i.branch() {
                    branches.push((out.len(), o.imm));
                }
                i.validate().map_err(|e| format!("{}: {e}", isa::disassemble_one(i)))?;
                out.push(i);
            }
        }
    }
    if out.last().is_none_or(|i| !matches!(i.op, op::S_ENDPGM | op::S_TRAP | op::S_BRANCH)) {
        labels.entry(u32::MAX).or_insert(out.len());
        out.push(Inst::new(op::S_ENDPGM));
    }
    for (pc, label) in branches {
        let t = *labels.get(&label).ok_or("undefined label")?;
        let t = t.min(out.len() - 1);
        out[pc].hi = (t as i64 - pc as i64 - 1) as i32 as u32;
    }
    Ok(out)
}

fn dump(title: &str, ops: &[Op]) {
    if std::env::var_os("APEX_DUMP_MIR").is_some() {
        eprintln!("-- {title}");
        for o in ops {
            let name = o.isa().and_then(|x| isa::NAMES.iter().find(|n| n.0 == x)).map(|n| n.1.to_ascii_lowercase());
            let name = name.unwrap_or_else(|| ["const", "copy", "label", "entry"][(o.op - 0x100) as usize].into());
            eprintln!("{name} {:?} hi=0x{:x} imm={}", o.f, o.hi, o.imm);
        }
    }
}

/// Rewrites every access of scalar `victim` through a temporary read from and
/// written to lanes `lane..` of the vector register `bank`. Lane moves ignore
/// `exec`, so the value survives regions where no lane is active.
fn spill_lanes(ops: &mut Vec<Op>, values: &mut Vec<Value>, align: &mut Vec<u8>, victim: u32, bank: u32, lane: u32) {
    let width = values[victim as usize].width;
    let mut out = Vec::with_capacity(ops.len() + 8);
    for mut o in ops.drain(..) {
        let access = o.access();
        let (mut reads, mut writes) = (false, false);
        for f in 0..4 {
            if o.f[f].val() == Some(victim) {
                match access[f] {
                    Some(Access::Def) => {
                        writes = true;
                        reads |= o.f[f].n() < width;
                    }
                    Some(Access::DefUse) => (reads, writes) = (true, true),
                    _ => reads = true,
                }
            }
        }
        if !reads && !writes {
            out.push(o);
            continue;
        }
        values.push(Value { class: Class::S, width });
        align.push(alignment(&values[victim as usize]));
        let t = values.len() as u32 - 1;
        for f in 0..4 {
            if let Opnd::Val { id, off, n } = o.f[f] {
                if id == victim {
                    o.f[f] = Opnd::Val { id: t, off, n };
                }
            }
        }
        let b = Opnd::Val { id: bank, off: 0, n: 1 };
        if reads {
            for k in 0..width {
                out.push(Op::new(op::V_READLANE as u16, [Opnd::Val { id: t, off: k, n: 1 }, b, Opnd::Lit(lane + k as u32), Opnd::None]));
            }
        }
        out.push(o);
        if writes {
            for k in 0..width {
                out.push(Op::new(op::V_WRITELANE as u16, [b, Opnd::Val { id: t, off: k, n: 1 }, Opnd::Lit(lane + k as u32), Opnd::None]));
            }
        }
    }
    *ops = out;
}

/// Rewrites every access of `victim` through a short-lived temporary loaded
/// from and stored to its private slot. Masked writes store only active lanes,
/// so the slot keeps the masked read-modify-write semantics of the register.
fn spill(ops: &mut Vec<Op>, values: &mut Vec<Value>, align: &mut Vec<u8>, victim: u32, slot: u32) {
    let width = values[victim as usize].width;
    let mut out = Vec::with_capacity(ops.len() + 8);
    // Private accesses move at most four dwords.
    let memory = |out: &mut Vec<Op>, op: u8, t: u32| {
        for k in (0..width).step_by(4) {
            let n = (width - k).min(4);
            out.push(Op {
                hi: (((slot + k as u32) * 4) & 0xfffff) | ((n as u32 - 1) << 20),
                ..Op::new(op as u16, [Opnd::Val { id: t, off: k, n }, Opnd::None, Opnd::None, Opnd::None])
            });
        }
    };
    for mut o in ops.drain(..) {
        let access = o.access();
        let mut reads = false;
        let mut writes = false;
        for f in 0..4 {
            if o.f[f].val() == Some(victim) {
                match access[f] {
                    Some(Access::Def) => {
                        writes = true;
                        reads |= o.f[f].n() < width;
                    }
                    Some(Access::DefUse) => {
                        reads = true;
                        writes = true;
                    }
                    _ => reads = true,
                }
            }
        }
        if !reads && !writes {
            out.push(o);
            continue;
        }
        values.push(Value { class: Class::V, width });
        align.push(if width == 2 { 2 } else { 1 });
        let t = values.len() as u32 - 1;
        for f in 0..4 {
            if let Opnd::Val { id, off, n } = o.f[f] {
                if id == victim {
                    o.f[f] = Opnd::Val { id: t, off, n };
                }
            }
        }
        if reads {
            memory(&mut out, op::SCRATCH_LOAD, t);
        }
        out.push(o);
        if writes {
            memory(&mut out, op::SCRATCH_STORE, t);
        }
    }
    *ops = out;
}

pub fn compile(mut ops: Vec<Op>, mut values: Vec<Value>, mut header: Program) -> Result<(Program, Stats), String> {
    check(&ops, &values)?;
    dump("input", &ops);
    fold_constants(&mut ops, &values);
    eliminate_dead(&mut ops);
    let mut fixed = bind_launch(&mut ops, &mut values);
    let align = coalesce(&mut ops, &values, &mut fixed);
    eliminate_dead(&mut ops);
    dump("coalesced", &ops);
    let mut ops = schedule::schedule(ops, &values);
    let mut align = align;
    let mut spilled = 0u32;
    // Compute launches spill vector values to private memory; scalars spill
    // into lanes of vector registers in every stage.
    let mut temps = std::collections::BTreeSet::new();
    let (mut lanes, mut bank, mut stats_lanes) = (0u32, 0u32, 0u32);
    let compute = header.stage == isa::Stage::Compute;
    let (homes, mut stats) = loop {
        match allocate(&ops, &values, &fixed, &align, Some(&temps)) {
            Ok(result) => break result,
            Err(p) if p.live.is_empty() => return Err(p.message + " without spilling"),
            Err(p) => {
                for victim in p.live {
                    let first = values.len() as u32;
                    let width = values[victim as usize].width as u32;
                    if values[victim as usize].class == Class::S {
                        // Scalars spill into lanes of vector registers.
                        let lane = lanes % 16;
                        if lane == 0 || lane + width > 16 {
                            values.push(Value { class: Class::V, width: 1 });
                            align.push(1);
                            lanes = lanes.next_multiple_of(16);
                            bank = values.len() as u32 - 1;
                            temps.insert(bank);
                        }
                        spill_lanes(&mut ops, &mut values, &mut align, victim, bank, lanes % 16);
                        lanes += width;
                        stats_lanes += width;
                    } else {
                        if !compute {
                            return Err("vector register pressure exceeds 128 registers without spilling".into());
                        }
                        let slot = header.private / 4 + spilled;
                        spilled += width;
                        spill(&mut ops, &mut values, &mut align, victim, slot);
                    }
                    temps.extend(first..values.len() as u32);
                }
            }
        }
    };
    stats.spills = spilled + stats_lanes;
    header.private += 4 * spilled;
    header.code = lower(&ops, &values, &homes)?;
    stats.instructions = header.code.len() as u32;
    header.validate()?;
    Ok((header, stats))
}
