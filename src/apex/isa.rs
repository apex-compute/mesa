// SPDX-License-Identifier: MIT
use std::collections::BTreeSet;

// Revision-1 provisional pipeline profile; change only with matching RTL evidence.
pub const INTEGER_MUL_LATENCY: usize = 4;
pub const FP32_LATENCY: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    S,
    V,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reg(pub Class, pub u8);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inst {
    pub op: u8,
    pub d: u8,
    pub a: u8,
    pub b: u8,
    pub c: u8,
    pub imm: u32,
}
impl Inst {
    pub fn new(op: u8) -> Self {
        Self {
            op,
            d: 0,
            a: 0,
            b: 0,
            c: 0,
            imm: 0,
        }
    }
    pub fn asynchronous(self) -> bool {
        (0x50..=0x59).contains(&self.op)
    }
    pub fn control(self) -> bool {
        matches!(self.op, 1 | 2 | 4 | 5 | 7)
    }
    pub fn latency(self) -> usize {
        match self.op {
            0x14 | 0x24 => INTEGER_MUL_LATENCY,
            0x30..=0x34 => FP32_LATENCY,
            _ => 1,
        }
    }
    // Field roles: class and number of consecutive words, or None for reserved.
    pub fn roles(self) -> Result<[Option<(Class, u8)>; 4], String> {
        use Class::*;
        let s = Some((S, 1));
        let v = Some((V, 1));
        let sp = Some((S, 2));
        let vp = Some((V, 2));
        Ok(match self.op {
            0..=4 | 7 => [None; 4],
            5 => [None, s, None, None],
            6 => [s, s, None, None],
            0x10 => [s, None, None, None],
            0x20 | 0x40 => [v, None, None, None],
            0x11 => [s, s, None, None],
            0x21 | 0x33 | 0x34 => [v, v, None, None],
            0x12..=0x1b => [s, s, s, None],
            0x22..=0x2b | 0x30 | 0x31 | 0x44 => [v, v, v, None],
            0x1c => [sp, sp, sp, None],
            0x2c => [vp, vp, vp, None],
            0x2d => [v, s, None, None],
            0x2e | 0x32 => [v, v, v, v],
            0x41 => [s, None, None, None],
            0x42 => [sp, None, None, None],
            0x43 => [s, v, None, None],
            0x50 => [v, vp, None, None],
            0x51 => [None, vp, v, None],
            0x52 => [v, vp, if self.imm == 2 { vp } else { v }, None],
            0x53 => [v, v, None, None],
            0x54 => [None, v, v, None],
            0x55 => [v, v, if self.imm == 2 { vp } else { v }, None],
            0x56 => [v, None, None, None],
            0x57 => [None, None, v, None],
            0x58 => [v, v, None, None],
            0x59 => [None, v, v, None],
            _ => return Err(format!("unknown opcode {:02x}", self.op)),
        })
    }
    pub fn regs(self, writes: bool) -> Vec<Reg> {
        let fields = [self.d, self.a, self.b, self.c];
        self.roles()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i == 0) == writes)
            .flat_map(|(i, r)| {
                r.into_iter()
                    .flat_map(move |&(cl, n)| (0..n).map(move |j| Reg(cl, fields[i] + j)))
            })
            .collect()
    }
    pub fn validate(self) -> Result<(), String> {
        let fields = [self.d, self.a, self.b, self.c];
        for (i, role) in self.roles()?.iter().enumerate() {
            if let Some((_, n)) = role {
                if fields[i] > 64 - n || (*n == 2 && fields[i] % 2 != 0) {
                    return Err("register/pair out of range".into());
                }
            } else if !(i == 3 && self.asynchronous()) && fields[i] != 0 {
                return Err("reserved register field".into());
            }
        }
        let valid = match self.op {
            2 | 4 | 5 | 0x10 | 0x20 | 0x56..=0x59 => true,
            3 => self.imm > 0 && self.imm < 16,
            0x40 => self.imm <= 4,
            0x41 => self.imm <= 6,
            0x42 => self.imm <= 1,
            0x52 | 0x55 => self.imm <= 9,
            _ => self.imm == 0,
        };
        if !valid || (self.asynchronous() && self.c >= 4) {
            return Err("reserved immediate/token".into());
        }
        Ok(())
    }
    pub fn encode(self) -> Result<u64, String> {
        self.validate()?;
        Ok(self.op as u64
            | (self.d as u64) << 8
            | (self.a as u64) << 14
            | (self.b as u64) << 20
            | (self.c as u64) << 26
            | (self.imm as u64) << 32)
    }
    pub fn decode(x: u64) -> Result<Self, String> {
        let i = Self {
            op: x as u8,
            d: ((x >> 8) & 63) as u8,
            a: ((x >> 14) & 63) as u8,
            b: ((x >> 20) & 63) as u8,
            c: ((x >> 26) & 63) as u8,
            imm: (x >> 32) as u32,
        };
        i.validate()?;
        Ok(i)
    }
    pub fn bank_legal(self) -> bool {
        let regs: BTreeSet<_> = self.regs(false).into_iter().collect();
        [Class::S, Class::V].iter().all(|cl| {
            (0..4).all(|b| regs.iter().filter(|r| r.0 == *cl && r.1 % 4 == b).count() <= 2)
        })
    }
}

pub const NAMES: &[(u8, &str)] = &[
    (0, "nop"),
    (1, "end"),
    (2, "trap"),
    (3, "wait"),
    (4, "branch"),
    (5, "branch_nz"),
    (6, "mask"),
    (7, "barrier"),
    (0x10, "s_imm"),
    (0x11, "s_mov"),
    (0x12, "s_add"),
    (0x13, "s_sub"),
    (0x14, "s_mul"),
    (0x15, "s_and"),
    (0x16, "s_or"),
    (0x17, "s_xor"),
    (0x18, "s_shl"),
    (0x19, "s_shr"),
    (0x1a, "s_lt"),
    (0x1b, "s_eq"),
    (0x1c, "s_add64"),
    (0x20, "v_imm"),
    (0x21, "v_mov"),
    (0x22, "v_add"),
    (0x23, "v_sub"),
    (0x24, "v_mul"),
    (0x25, "v_and"),
    (0x26, "v_or"),
    (0x27, "v_xor"),
    (0x28, "v_shl"),
    (0x29, "v_shr"),
    (0x2a, "v_lt"),
    (0x2b, "v_eq"),
    (0x2c, "v_add64"),
    (0x2d, "broadcast"),
    (0x2e, "select"),
    (0x30, "fadd"),
    (0x31, "fmul"),
    (0x32, "ffma"),
    (0x33, "u2f"),
    (0x34, "f2u"),
    (0x40, "identity"),
    (0x41, "launch"),
    (0x42, "clock"),
    (0x43, "ballot"),
    (0x44, "shuffle"),
    (0x50, "load"),
    (0x51, "store"),
    (0x52, "atomic_add"),
    (0x53, "shared_load"),
    (0x54, "shared_store"),
    (0x55, "shared_atomic_add"),
    (0x56, "private_load"),
    (0x57, "private_store"),
    (0x58, "private_load_index"),
    (0x59, "private_store_index"),
];

pub fn assemble(text: &str) -> Result<Vec<Inst>, String> {
    text.lines()
        .filter(|s| !s.trim().is_empty())
        .map(|line| {
            let p: Vec<_> = line.split_whitespace().collect();
            if p.len() != 6 {
                return Err("expected mnemonic dst src0 src1 src2 immediate".into());
            }
            let op = NAMES
                .iter()
                .find(|(_, n)| *n == p[0])
                .ok_or("unknown mnemonic")?
                .0;
            let num = |s: &str| -> Result<u32, String> {
                if let Some(x) = s.strip_prefix("0x") {
                    u32::from_str_radix(x, 16)
                } else {
                    s.parse()
                }
                .map_err(|_| "invalid integer".into())
            };
            let r = |s| -> Result<u8, String> {
                u8::try_from(num(s)?).map_err(|_| "register overflow".into())
            };
            let i = Inst {
                op,
                d: r(p[1])?,
                a: r(p[2])?,
                b: r(p[3])?,
                c: r(p[4])?,
                imm: num(p[5])?,
            };
            i.validate()?;
            Ok(i)
        })
        .collect()
}
pub fn disassemble(code: &[Inst]) -> String {
    code.iter()
        .map(|i| {
            format!(
                "{} {} {} {} {} 0x{:08x}\n",
                NAMES.iter().find(|(op, _)| *op == i.op).unwrap().1,
                i.d,
                i.a,
                i.b,
                i.c,
                i.imm
            )
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Program {
    pub code: Vec<Inst>,
    pub entry: u32,
    pub scalar: u32,
    pub vector: u32,
    pub shared: u32,
    pub private: u32,
}
impl Program {
    pub fn validate(&self) -> Result<(), String> {
        if self.code.is_empty()
            || self.code.len() > u32::MAX as usize
            || self.entry as usize >= self.code.len()
            || self.scalar > 64
            || self.vector > 64
            || self.shared > 32768
            || self.private % 4 != 0
        {
            return Err("invalid launch metadata".into());
        }
        for i in &self.code {
            i.validate()?;
            if matches!(i.op, 4 | 5) && i.imm as usize >= self.code.len() {
                return Err("branch out of code".into());
            }
            if matches!(i.op, 0x56..=0x59) && i.imm >= self.private / 4 {
                return Err("private word out of allocation".into());
            }
            for Reg(c, r) in i.regs(false).into_iter().chain(i.regs(true)) {
                if r as u32
                    >= if c == Class::S {
                        self.scalar
                    } else {
                        self.vector
                    }
                {
                    return Err("register exceeds metadata".into());
                }
            }
            if !i.bank_legal() {
                return Err("register read bank conflict".into());
            }
        }
        Ok(())
    }
    pub fn bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let h = [
            0x31585041,
            1,
            self.code.len() as u32,
            self.entry,
            self.scalar,
            self.vector,
            self.shared,
            self.private,
            16,
            4,
        ];
        let mut out: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
        for i in &self.code {
            out.extend(i.encode()?.to_le_bytes())
        }
        Ok(out)
    }
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < 40 {
            return Err("short header".into());
        }
        let h: Vec<u32> = b[..40]
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        if h[0] != 0x31585041
            || h[1] != 1
            || h[8] != 16
            || h[9] != 4
            || (h[2] as u64) * 8 + 40 != b.len() as u64
        {
            return Err("invalid container".into());
        }
        let code = b[40..]
            .chunks_exact(8)
            .map(|v| Inst::decode(u64::from_le_bytes(v.try_into().unwrap())))
            .collect::<Result<_, _>>()?;
        let p = Self {
            code,
            entry: h[3],
            scalar: h[4],
            vector: h[5],
            shared: h[6],
            private: h[7],
        };
        p.validate()?;
        Ok(p)
    }
}
pub fn private_address(
    base: u64,
    waves: u64,
    words: u32,
    wave: u64,
    word: u32,
    lane: u32,
    backing: u64,
) -> Result<u64, String> {
    let size = waves
        .checked_mul(words as u64)
        .and_then(|v| v.checked_mul(64))
        .ok_or("private allocation overflow")?;
    if size > backing
        || wave >= waves
        || word >= words
        || lane >= 16
        || base.checked_add(size).is_none()
    {
        return Err("private allocation bounds".into());
    }
    Ok(base + ((wave * words as u64 + word as u64) * 16 + lane as u64) * 4)
}
