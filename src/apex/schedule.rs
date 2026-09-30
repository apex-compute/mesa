// SPDX-License-Identifier: MIT
// Pre-allocation list scheduling inside straight-line regions. The core's
// scoreboard resolves every hazard, so the schedule only orders for latency
// within register pressure and never pads.
use crate::isa::{op, Class};
use crate::mir::{Op, Value, REORDER};
use std::collections::BTreeMap;

/// Issue-to-dependent latency in cycles (architecture: FMA-class results reach
/// a dependent about 12 cycles later; loads and texture use nominal hits).
pub fn latency(o: &Op) -> u32 {
    let Some(x) = o.isa() else { return 1 };
    match x {
        op::V_MUL_LO..=op::V_MUL_HI_I => 6,
        op::V_ADD_F..=op::V_LDEXP | op::V_CVT_F_U..=op::V_CUBEMA | op::V_INTERP | op::V_INTERP_FLAT => 12,
        op::V_RCP..=op::V_COS => 16,
        op::V_CMP_F | op::V_CMP_CLASS => 12,
        op::V_READLANE..=op::V_MBCNT => 4,
        0x60..=0xab => 4,
        op::S_LOAD | op::S_BUFFER_LOAD => 24,
        op::GLOBAL_LOAD | op::BUFFER_LOAD | op::SCRATCH_LOAD => 80,
        op::SHARED_LOAD | op::SHARED_ATOMIC => 24,
        op::IMAGE_SAMPLE | op::IMAGE_FETCH => 60,
        _ => 2,
    }
}

/// Registers above which a candidate that grows pressure waits.
const VECTOR_PRESSURE: i32 = 96;
const SCALAR_PRESSURE: i32 = 64;

pub fn schedule(ops: Vec<Op>, values: &[Value]) -> Vec<Op> {
    let mut out = Vec::with_capacity(ops.len());
    let mut region = Vec::new();
    for o in ops {
        if o.boundary() {
            schedule_region(std::mem::take(&mut region), values, &mut out);
            out.push(o);
        } else {
            region.push(o);
        }
    }
    schedule_region(region, values, &mut out);
    out
}

fn schedule_region(ops: Vec<Op>, values: &[Value], out: &mut Vec<Op>) {
    let n = ops.len();
    if n < 2 {
        out.extend(ops);
        return;
    }
    let mut preds: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
    let mut last_def: BTreeMap<u32, usize> = BTreeMap::new();
    let mut readers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    let mut last_side: Option<usize> = None;
    let mut loads_since: Vec<usize> = Vec::new();
    for (i, o) in ops.iter().enumerate() {
        for u in o.uses() {
            if let Some(&d) = last_def.get(&u) {
                preds[i].push((d, latency(&ops[d])));
            }
        }
        for d in o.defs() {
            if let Some(&p) = last_def.get(&d) {
                preds[i].push((p, 1));
            }
            for &r in readers.get(&d).into_iter().flatten() {
                if r != i {
                    preds[i].push((r, 0));
                }
            }
        }
        let memory = o.side_effect() || (o.load() && o.flags & REORDER == 0);
        if memory {
            if let Some(s) = last_side {
                preds[i].push((s, 0));
            }
            if o.side_effect() {
                for &l in &loads_since {
                    preds[i].push((l, 0));
                }
                loads_since.clear();
                last_side = Some(i);
            } else {
                loads_since.push(i);
            }
        }
        for u in o.uses() {
            readers.entry(u).or_default().push(i);
        }
        for d in o.defs() {
            last_def.insert(d, i);
            readers.remove(&d);
        }
    }
    let mut succs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
    for (i, p) in preds.iter().enumerate() {
        for &(j, l) in p {
            succs[j].push((i, l));
        }
    }
    // Critical path to the region end.
    let mut height = vec![0u32; n];
    for i in (0..n).rev() {
        height[i] = latency(&ops[i]) + succs[i].iter().map(|&(s, l)| height[s] + l.saturating_sub(latency(&ops[i]))).max().unwrap_or(0);
    }
    // Remaining uses per value inside the region, for pressure.
    let mut remaining: BTreeMap<u32, usize> = BTreeMap::new();
    for o in &ops {
        for u in o.uses() {
            *remaining.entry(u).or_default() += 1;
        }
    }
    let class = |id: u32| (values[id as usize].class == Class::V) as usize;
    let width = |id: u32| values[id as usize].width as i32;
    let mut live: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut pressure = [0i32; 2];
    let mut waiting: Vec<usize> = preds.iter().map(|p| p.len()).collect();
    let mut ready_at = vec![0u32; n];
    let mut ready: Vec<usize> = (0..n).filter(|&i| waiting[i] == 0).collect();
    let mut cycle = 0u32;
    let mut slots: Vec<Option<Op>> = ops.into_iter().map(Some).collect();
    while !ready.is_empty() {
        let delta = |i: usize, k: usize, slots: &Vec<Option<Op>>, remaining: &BTreeMap<u32, usize>, live: &std::collections::BTreeSet<u32>| -> i32 {
            let o = slots[i].as_ref().unwrap();
            let grow: i32 = o.defs().filter(|&d| class(d) == k && !live.contains(&d)).map(width).sum();
            let free: i32 = o.uses().filter(|&u| class(u) == k && remaining.get(&u) == Some(&1)).map(width).sum();
            grow - free
        };
        let pick = *ready
            .iter()
            .max_by_key(|&&i| {
                let limits = [SCALAR_PRESSURE, VECTOR_PRESSURE];
                let over = (0..2).map(|k| {
                    let d = delta(i, k, &slots, &remaining, &live);
                    (pressure[k] + d > limits[k] && d > 0) as u32
                }).sum::<u32>();
                (std::cmp::Reverse(over), ready_at[i] <= cycle, height[i], std::cmp::Reverse(i))
            })
            .unwrap();
        ready.retain(|&i| i != pick);
        let o = slots[pick].take().unwrap();
        cycle = cycle.max(ready_at[pick]) + 1;
        for u in o.uses() {
            let r = remaining.get_mut(&u).unwrap();
            *r -= 1;
            if *r == 0 && live.remove(&u) {
                pressure[class(u)] -= width(u);
            }
        }
        for d in o.defs() {
            if remaining.get(&d).is_some_and(|&r| r > 0) && live.insert(d) {
                pressure[class(d)] += width(d);
            }
        }
        for &(s, l) in &succs[pick] {
            ready_at[s] = ready_at[s].max(cycle + l);
            waiting[s] -= 1;
            if waiting[s] == 0 {
                ready.push(s);
            }
        }
        out.push(o);
    }
}
