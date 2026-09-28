// SPDX-License-Identifier: MIT
use crate::isa::{Inst, Reg};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
struct Scheduler {
    out: Vec<Inst>,
    ready: BTreeMap<Reg, usize>,
    tokens: [Option<Vec<Reg>>; 4],
    stores: u32,
    // Cycle whose edge writes each scalar register (see scalar_port1_reads).
    edges: BTreeMap<Reg, usize>,
}
impl Scheduler {
    fn live(&self) -> u32 {
        self.tokens
            .iter()
            .enumerate()
            .fold(0, |v, (t, r)| v | if r.is_some() { 1 << t } else { 0 })
    }
    fn wait(&mut self, mask: u32) {
        if mask == 0 {
            return;
        }
        self.out.push(Inst {
            imm: mask,
            ..Inst::new(3)
        });
        for t in 0..4 {
            if mask & (1 << t) != 0 {
                self.tokens[t] = None;
            }
        }
        self.stores &= !mask;
    }
    fn drain(&mut self) {
        self.wait(self.live());
        let end = self.ready.values().copied().max().unwrap_or(0);
        // Past every scalar write edge's port-1 hazard cycle as well.
        let edge = self.edges.values().map(|e| e + 2).max().unwrap_or(0);
        while self.out.len() < end.max(edge) {
            self.out.push(Inst::new(0));
        }
        self.ready.clear();
        self.edges.clear();
    }
    fn issue(&mut self, mut i: Inst) {
        let reads = i.regs(false);
        let writes = i.regs(true);
        let touched: Vec<_> = reads.iter().chain(&writes).copied().collect();
        let mut wait = 0;
        for (t, r) in self.tokens.iter().enumerate() {
            if r.as_ref()
                .is_some_and(|r| r.iter().any(|r| touched.contains(r)))
            {
                wait |= 1 << t;
            }
        }
        // Preserve potentially aliasing memory order; independent reads overlap.
        if i.asynchronous() {
            if matches!(i.op, 0x51 | 0x52 | 0x54 | 0x55 | 0x57 | 0x59) {
                wait |= self.live();
            } else {
                wait |= self.stores;
            }
        }
        self.wait(wait);
        if i.control() || i.op == 6 {
            self.drain();
        }
        let ready = touched
            .iter()
            .filter_map(|r| self.ready.get(r))
            .copied()
            .max()
            .unwrap_or(0);
        while self.out.len() < ready {
            self.out.push(Inst::new(0));
        }
        let port1 = i.scalar_port1_reads();
        while port1.iter().any(|r| self.edges.get(r).is_some_and(|&e| e + 1 == self.out.len())) {
            self.out.push(Inst::new(0));
        }
        if i.asynchronous() {
            if self.live() == 15 {
                self.wait(1);
            }
            let t = self.tokens.iter().position(Option::is_none).unwrap();
            i.c = t as u8;
            self.tokens[t] = Some(writes.clone());
            if matches!(i.op, 0x51 | 0x52 | 0x54 | 0x55 | 0x57 | 0x59) {
                self.stores |= 1 << t;
            }
        } else {
            for r in writes {
                self.ready.insert(r, self.out.len() + i.latency());
                if r.0 == crate::isa::Class::S {
                    self.edges.insert(r, self.out.len() + i.latency());
                }
            }
        }
        self.out.push(i);
    }
}

pub fn schedule(input: &[Inst]) -> Result<Vec<Inst>, String> {
    let targets: BTreeSet<usize> = input
        .iter()
        .filter(|i| matches!(i.op, 4 | 5))
        .map(|i| i.imm as usize)
        .collect();
    if targets.iter().any(|&t| t >= input.len()) {
        return Err("branch target out of input".into());
    }
    let mut s = Scheduler::default();
    let mut map = vec![0; input.len()];
    let mut branches = Vec::new();
    for (pc, &i) in input.iter().enumerate() {
        i.validate()?;
        if i.op == 3 {
            return Err("MIR scheduler owns token waits".into());
        }
        if targets.contains(&pc) {
            s.drain();
        }
        map[pc] = s.out.len();
        if !i.bank_legal() {
            return Err("allocator left bank conflict".into());
        }
        s.issue(i);
        if matches!(i.op, 4 | 5) {
            branches.push(s.out.len() - 1);
        }
    }
    for pc in branches {
        s.out[pc].imm =
            u32::try_from(map[s.out[pc].imm as usize]).map_err(|_| "code size overflow")?;
    }
    Ok(s.out)
}

// Independent static schedule checker. Branch boundaries must be quiescent;
// thus the linear check also covers backward edges without assuming trip counts.
pub fn validate(code: &[Inst]) -> Result<(), String> {
    let targets: BTreeSet<_> = code
        .iter()
        .filter(|i| matches!(i.op, 4 | 5))
        .map(|i| i.imm as usize)
        .collect();
    if targets.iter().any(|&t| t >= code.len())
        || code.last().is_none_or(|i| !matches!(i.op, 1 | 2 | 4))
    {
        return Err("branch/fallthrough outside program".into());
    }
    let mut pending: BTreeMap<Reg, usize> = BTreeMap::new();
    let mut tokens: [Option<Vec<Reg>>; 4] = Default::default();
    for (cycle, &i) in code.iter().enumerate() {
        i.validate()?;
        if !i.bank_legal() {
            return Err("bank over-subscription".into());
        }
        let live = || tokens.iter().any(Option::is_some);
        if (i.control() || i.op == 6 || targets.contains(&cycle))
            && (live() || pending.values().any(|&r| r > cycle))
        {
            return Err(format!("non-quiescent boundary {cycle}"));
        }
        if i.op == 3 {
            for t in 0..4 {
                if i.imm & (1 << t) != 0 {
                    if tokens[t].take().is_none() {
                        return Err("wait on dead token".into());
                    }
                }
            }
            continue;
        }
        if i.scalar_port1_reads().iter().any(|r| pending.get(r).is_some_and(|&t| t + 1 == cycle)) {
            return Err(format!("scalar port-1 write-edge read at {cycle}"));
        }
        let touched: Vec<_> = i.regs(false).into_iter().chain(i.regs(true)).collect();
        for r in touched {
            if pending.get(&r).is_some_and(|&t| t > cycle) {
                return Err(format!("RAW/WAW at {cycle}"));
            }
            if tokens.iter().flatten().any(|v| v.contains(&r)) {
                return Err("unwaited async result".into());
            }
        }
        if i.asynchronous() {
            if tokens[i.c as usize].is_some() {
                return Err("token reuse".into());
            }
            tokens[i.c as usize] = Some(i.regs(true));
        } else {
            for r in i.regs(true) {
                pending.insert(r, cycle + i.latency());
            }
        }
    }
    if tokens.iter().any(Option::is_some) {
        return Err("live tokens at code end".into());
    }
    Ok(())
}
