// SPDX-License-Identifier: MIT
// Executable model of the P7 ISA for compiler tests. It follows the
// architecture contract and records the choices the contract leaves open in
// isa-notes.md; the RTL reference simulator decides at integration.
use crate::isa::{self, op, Format, Inst, Program, Stage, EXEC, LANE, LITERAL, SCALAR};

pub struct Region<'a> {
    pub gpuva: u64,
    pub data: &'a mut [u8],
}
pub struct Memory<'a> {
    pub regions: Vec<Region<'a>>,
}
impl<'a> Memory<'a> {
    fn find(&mut self, address: u64, bytes: u64) -> Result<&mut [u8], String> {
        for r in &mut self.regions {
            if address >= r.gpuva && address + bytes <= r.gpuva + r.data.len() as u64 {
                let o = (address - r.gpuva) as usize;
                return Ok(&mut r.data[o..o + bytes as usize]);
            }
        }
        Err(format!("memory fault at 0x{address:x}"))
    }
    pub fn read(&mut self, address: u64) -> Result<u32, String> {
        if address % 4 != 0 {
            return Err(format!("unaligned access at 0x{address:x}"));
        }
        Ok(u32::from_le_bytes(self.find(address, 4)?.try_into().unwrap()))
    }
    pub fn write(&mut self, address: u64, value: u32) -> Result<(), String> {
        if address % 4 != 0 {
            return Err(format!("unaligned access at 0x{address:x}"));
        }
        self.find(address, 4)?.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}

/// Launch state of one vertex or fragment wave; compute waves derive theirs.
#[derive(Clone)]
pub struct WaveInit {
    pub exec: u16,
    pub scalar: [u32; 32],
    pub vector: [[u32; 16]; 72],
    /// Fragment lanes' primitive slots, kept by the core beside the wave.
    pub primitive: [u8; 16],
    /// Attribute blocks: [primitive][component][P0, P10, P20], component = input · 4 + c.
    pub attributes: Vec<[[f32; 3]; 144]>,
}
#[derive(Clone, Default)]
pub struct Exports {
    pub values: [[[u32; 4]; 16]; 10],
    pub lanes: [u16; 10],
    pub done: bool,
}
#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub instructions: u64,
    pub vector: u64,
    pub scalar: u64,
    pub memory: u64,
}

struct Wave {
    v: Vec<[u32; 16]>,
    s: [u32; 96],
    exec: u16,
    launch: u16,
    pc: usize,
    done: bool,
    barrier: bool,
    index: u32,
    group: [u32; 3],
    primitive: [u8; 16],
}

pub struct Machine<'m, 'a> {
    pub program: &'m Program,
    pub memory: &'m mut Memory<'a>,
    pub attributes: Vec<[[f32; 3]; 144]>,
    pub exports: Exports,
    pub stats: Stats,
    shared: Vec<u8>,
    pub limit: u64,
}

fn f(x: u32) -> f32 {
    f32::from_bits(x)
}
fn canonical(x: f32) -> u32 {
    if x.is_nan() {
        0x7fc0_0000
    } else {
        x.to_bits()
    }
}
fn min_f(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == b {
        if a.is_sign_negative() { a } else { b }
    } else if a < b {
        a
    } else {
        b
    }
}
fn max_f(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a == b {
        if a.is_sign_negative() { b } else { a }
    } else if a > b {
        a
    } else {
        b
    }
}
fn bfe(a: u32, b: u32, signed: bool) -> u32 {
    let (offset, width) = (b & 31, (b >> 8) & 63);
    if width == 0 {
        return 0;
    }
    let field = ((a as u64) >> offset) & if width >= 32 { u32::MAX as u64 } else { (1 << width) - 1 };
    if signed && width < 32 && (field >> (width - 1)) & 1 != 0 {
        (field | !((1u64 << width) - 1)) as u32
    } else {
        field as u32
    }
}
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    let (m, shift) = if e <= 0 {
        if e < -10 {
            return sign;
        }
        (man | 0x80_0000, (14 - e) as u32)
    } else {
        (man, 13)
    };
    let mut h = (m >> shift) as u32;
    let rest = m & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rest > half || (rest == half && h & 1 != 0) {
        h += 1;
    }
    if e > 0 {
        (sign as u32 | ((e as u32) << 10) + h) as u16
    } else {
        sign | h as u16
    }
}
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 31) as i32;
    let m = (h & 0x3ff) as f32;
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}
fn cube(x: f32, y: f32, z: f32) -> (f32, f32, f32, f32) {
    // Face id, sc, tc, 2·major axis (the AMD-style cube helpers).
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    if az >= ax && az >= ay {
        (if z < 0.0 { 5.0 } else { 4.0 }, if z < 0.0 { -x } else { x }, -y, 2.0 * z)
    } else if ay >= ax {
        (if y < 0.0 { 3.0 } else { 2.0 }, x, if y < 0.0 { -z } else { z }, 2.0 * y)
    } else {
        (if x < 0.0 { 1.0 } else { 0.0 }, if x < 0.0 { z } else { -z }, -y, 2.0 * x)
    }
}

impl<'m, 'a> Machine<'m, 'a> {
    pub fn new(program: &'m Program, memory: &'m mut Memory<'a>) -> Self {
        Self {
            program,
            memory,
            attributes: Vec::new(),
            exports: Exports::default(),
            stats: Stats::default(),
            shared: Vec::new(),
            limit: 50_000_000,
        }
    }

    /// Runs every workgroup of a compute grid; user data fills s0-s15.
    pub fn compute(&mut self, user: &[u32; 16], groups: [u32; 3], private_base: u64) -> Result<(), String> {
        if self.program.stage != Stage::Compute {
            return Err("compute launch of a graphics program".into());
        }
        let invocations = self.program.invocations();
        let count = invocations.div_ceil(16);
        for z in 0..groups[2] {
            for y in 0..groups[1] {
                for x in 0..groups[0] {
                    self.shared = vec![0; self.program.shared as usize];
                    let mut waves: Vec<Wave> = (0..count)
                        .map(|w| {
                            let lanes = (invocations - w * 16).min(16);
                            let mut wave = Wave::new(if lanes == 16 { 0xffff } else { (1 << lanes) - 1 });
                            wave.s[..16].copy_from_slice(user);
                            wave.s[16] = x;
                            wave.s[17] = y;
                            wave.s[18] = z;
                            wave.s[19] = w;
                            wave.s[20] = private_base as u32;
                            wave.s[21] = (private_base >> 32) as u32;
                            wave.index = w;
                            wave.group = [x, y, z];
                            for l in 0..16 {
                                wave.v[0][l] = w * 16 + l as u32;
                            }
                            wave
                        })
                        .collect();
                    self.run(&mut waves)?;
                }
            }
        }
        Ok(())
    }

    /// Runs one vertex or fragment wave from explicit launch registers.
    pub fn wave(&mut self, init: &WaveInit) -> Result<(), String> {
        if self.program.stage == Stage::Compute {
            return Err("single-wave launch of a compute program".into());
        }
        let mut w = Wave::new(init.exec);
        w.s[..32].copy_from_slice(&init.scalar);
        for r in 0..72 {
            w.v[r] = init.vector[r];
        }
        w.primitive = init.primitive;
        self.attributes = init.attributes.clone();
        self.run(&mut [w])?;
        if self.program.stage == Stage::Fragment && !self.exports.done {
            return Err("fragment program ended without a done export".into());
        }
        Ok(())
    }

    fn run(&mut self, waves: &mut [Wave]) -> Result<(), String> {
        for w in waves.iter_mut() {
            w.pc = self.program.entry as usize;
        }
        loop {
            let mut progress = false;
            for i in 0..waves.len() {
                if waves[i].done || waves[i].barrier {
                    continue;
                }
                self.step(&mut waves[i])?;
                progress = true;
            }
            if waves.iter().all(|w| w.done) {
                return Ok(());
            }
            if waves.iter().all(|w| w.done || w.barrier) {
                for w in waves.iter_mut() {
                    w.barrier = false;
                }
            } else if !progress {
                return Err("deadlock".into());
            }
            if self.stats.instructions > self.limit {
                return Err("instruction limit".into());
            }
        }
    }

    fn scalar(&self, w: &Wave, code: u8, literal: u32) -> Result<u32, String> {
        Ok(match code {
            SCALAR..=223 => w.s[(code - SCALAR) as usize],
            EXEC => w.exec as u32,
            240..=254 => isa::INLINE[(code - 240) as usize],
            LITERAL => literal,
            _ => return Err(format!("operand {code} is not scalar")),
        })
    }
    fn lanes(&self, w: &Wave, code: u8, literal: u32) -> Result<[u32; 16], String> {
        Ok(match code {
            0..=127 => w.v[code as usize],
            LANE => std::array::from_fn(|l| l as u32),
            _ => [self.scalar(w, code, literal)?; 16],
        })
    }
    fn sgroup(&self, w: &Wave, code: u8, n: usize) -> Vec<u32> {
        let base = (code - SCALAR) as usize;
        w.s[base..base + n].to_vec()
    }

    fn step(&mut self, w: &mut Wave) -> Result<(), String> {
        let i = *self.program.code.get(w.pc).ok_or("pc outside the program")?;
        self.stats.instructions += 1;
        let fmt = isa::format(i.op).unwrap();
        let next = w.pc + 1;
        w.pc = next;
        let literal = i.hi;
        let branch = |w: &mut Wave, taken: bool| {
            if taken {
                w.pc = (next as i64 + i.hi as i32 as i64) as usize;
            }
        };
        match fmt {
            Format::Control => {
                self.stats.scalar += 1;
                match i.op {
                    op::S_NOP | op::S_FENCE | op::S_SLEEP => {}
                    op::S_ENDPGM => w.done = true,
                    op::S_TRAP => return Err(format!("trap {}", i.hi)),
                    op::S_BRANCH => branch(w, true),
                    op::S_CBRANCH_Z => {
                        let v = self.scalar(w, i.a, 0)?;
                        branch(w, v == 0)
                    }
                    op::S_CBRANCH_NZ => {
                        let v = self.scalar(w, i.a, 0)?;
                        branch(w, v != 0)
                    }
                    op::S_CBRANCH_EXECZ => branch(w, w.exec == 0),
                    op::S_CBRANCH_EXECNZ => branch(w, w.exec != 0),
                    op::S_BARRIER => {
                        if self.program.stage != Stage::Compute {
                            return Err("barrier outside compute".into());
                        }
                        w.barrier = true
                    }
                    _ => unreachable!(),
                }
            }
            Format::Salu => {
                self.stats.scalar += 1;
                self.salu(w, i, literal)?
            }
            Format::Valu => {
                self.stats.vector += 1;
                self.valu(w, i, literal)?
            }
            Format::Memory => {
                self.stats.memory += 1;
                self.memory_op(w, i)?
            }
            Format::Texture => {
                self.stats.memory += 1;
                self.texture(w, i)?
            }
            Format::Export => {
                if self.program.stage != Stage::Fragment || self.exports.done {
                    return Err("export outside a fragment program or after done".into());
                }
                let target = i.d as usize;
                for l in 0..16 {
                    if w.exec >> l & 1 != 0 {
                        for c in 0..4 {
                            self.exports.values[target][l][c] = w.v[i.a as usize + c][l];
                        }
                    }
                }
                self.exports.lanes[target] |= w.exec;
                if i.b & 1 != 0 {
                    self.exports.done = true;
                }
            }
        }
        Ok(())
    }

    fn salu(&mut self, w: &mut Wave, i: Inst, literal: u32) -> Result<(), String> {
        let read = |m: &Self, w: &Wave, code: u8| m.scalar(w, code, literal);
        let pair = |w: &Wave, code: u8| -> u64 {
            let r = (code - SCALAR) as usize;
            w.s[r] as u64 | (w.s[r + 1] as u64) << 32
        };
        let write = |w: &mut Wave, value: u32| {
            if i.d == EXEC {
                w.exec = value as u16;
            } else {
                w.s[(i.d - SCALAR) as usize] = value;
            }
        };
        let write64 = |w: &mut Wave, value: u64| {
            let r = (i.d - SCALAR) as usize;
            w.s[r] = value as u32;
            w.s[r + 1] = (value >> 32) as u32;
        };
        match i.op {
            op::S_ADD64 => {
                let v = pair(w, i.a).wrapping_add(pair(w, i.b));
                write64(w, v);
                return Ok(());
            }
            op::S_SUB64 => {
                let v = pair(w, i.a).wrapping_sub(pair(w, i.b));
                write64(w, v);
                return Ok(());
            }
            op::S_SHL64 => {
                let v = pair(w, i.a) << (read(self, w, i.b)? & 63);
                write64(w, v);
                return Ok(());
            }
            op::S_MEMTIME => {
                write64(w, self.stats.instructions);
                return Ok(());
            }
            op::S_SETEXEC => {
                w.exec = read(self, w, i.a)? as u16 & w.launch;
                return Ok(());
            }
            _ => {}
        }
        let a = read(self, w, i.a)?;
        let b = if matches!(i.op, op::S_MOV | op::S_NOT | op::S_FF1 | op::S_POPCNT | op::S_AND_SAVEEXEC..=op::S_LAUNCH) {
            0
        } else {
            read(self, w, i.b)?
        };
        let exec = w.exec as u32;
        let v = match i.op {
            op::S_MOV => a,
            op::S_ADD => a.wrapping_add(b),
            op::S_SUB => a.wrapping_sub(b),
            op::S_MUL => a.wrapping_mul(b),
            op::S_MUL_HI_U => ((a as u64 * b as u64) >> 32) as u32,
            op::S_AND => a & b,
            op::S_OR => a | b,
            op::S_XOR => a ^ b,
            op::S_ANDN2 => a & !b,
            op::S_ORN2 => a | !b,
            op::S_NOT => !a,
            op::S_SHL => a << (b & 31),
            op::S_SHR => a >> (b & 31),
            op::S_ASHR => ((a as i32) >> (b & 31)) as u32,
            op::S_BFE_U => bfe(a, b, false),
            op::S_BFE_I => bfe(a, b, true),
            op::S_MIN_I => (a as i32).min(b as i32) as u32,
            op::S_MIN_U => a.min(b),
            op::S_MAX_I => (a as i32).max(b as i32) as u32,
            op::S_MAX_U => a.max(b),
            op::S_CMP_EQ => (a == b) as u32,
            op::S_CMP_NE => (a != b) as u32,
            op::S_CMP_LT_I => ((a as i32) < (b as i32)) as u32,
            op::S_CMP_LT_U => (a < b) as u32,
            op::S_CMP_LE_I => ((a as i32) <= (b as i32)) as u32,
            op::S_CMP_LE_U => (a <= b) as u32,
            op::S_CSELECT => {
                if read(self, w, i.hi as u8)? != 0 { a } else { b }
            }
            op::S_FF1 => if a == 0 { u32::MAX } else { a.trailing_zeros() },
            op::S_POPCNT => a.count_ones(),
            op::S_AND_SAVEEXEC => {
                w.exec = (exec & a) as u16;
                exec
            }
            op::S_OR_SAVEEXEC => {
                w.exec = ((exec | a) as u16) & w.launch;
                exec
            }
            op::S_ANDN2_SAVEEXEC => {
                w.exec = (a & !exec) as u16;
                exec
            }
            op::S_LAUNCH => match a {
                0..=2 => w.group[a as usize],
                3 => w.index,
                4..=6 => self.program.local[a as usize - 4],
                _ => return Err("reserved launch value".into()),
            },
            _ => unreachable!(),
        };
        write(w, v);
        Ok(())
    }

    fn valu(&mut self, w: &mut Wave, i: Inst, literal: u32) -> Result<(), String> {
        let lit = i.literal();
        let fp = isa::fp_source(i.op);
        let modify = |x: u32, f: usize| -> u32 {
            if lit || !fp {
                return x;
            }
            let mut x = x;
            if (i.hi >> (10 + f)) & 1 != 0 {
                x &= 0x7fff_ffff;
            }
            if (i.hi >> (7 + f)) & 1 != 0 {
                x ^= 0x8000_0000;
            }
            x
        };
        let kinds = i.kinds()?;
        let get = |m: &Self, w: &Wave, f: usize| -> Result<[u32; 16], String> {
            if kinds[f] == isa::Kind::None || kinds[f] == isa::Kind::Raw {
                return Ok([0; 16]);
            }
            let mut v = m.lanes(w, i.field(f), literal)?;
            for x in &mut v {
                *x = modify(*x, f);
            }
            Ok(v)
        };
        let (a, b, c) = (get(self, w, 1)?, get(self, w, 2)?, if lit { [0; 16] } else { get(self, w, 3)? });
        let exec = w.exec;
        let active = |l: usize| exec >> l & 1 != 0;
        let cond = i.hi as u8;
        // Scalar-destination and cross-lane forms.
        match i.op {
            op::V_CMP_I | op::V_CMP_U | op::V_CMP_F | op::V_CMP_CLASS => {
                let mut mask = 0u32;
                for l in 0..16 {
                    let t = match i.op {
                        op::V_CMP_CLASS => {
                            let x = f(a[l]);
                            let class = if x.is_nan() {
                                if a[l] & 0x40_0000 != 0 { 1 } else { 0 }
                            } else if x.is_infinite() {
                                if x < 0.0 { 2 } else { 9 }
                            } else if x == 0.0 {
                                if x.is_sign_negative() { 5 } else { 6 }
                            } else if x.is_subnormal() {
                                if x < 0.0 { 4 } else { 7 }
                            } else if x < 0.0 {
                                3
                            } else {
                                8
                            };
                            b[l] >> class & 1 != 0
                        }
                        op::V_CMP_F => {
                            let (x, y) = (f(a[l]), f(b[l]));
                            match cond {
                                0 => x == y,
                                1 => x != y,
                                2 => x < y,
                                3 => x <= y,
                                4 => x > y,
                                5 => x >= y,
                                6 => !x.is_nan() && !y.is_nan(),
                                7 => x.is_nan() || y.is_nan(),
                                _ => return Err("reserved compare".into()),
                            }
                        }
                        _ => {
                            let (x, y) = if i.op == op::V_CMP_I {
                                (a[l] as i32 as i64, b[l] as i32 as i64)
                            } else {
                                (a[l] as i64, b[l] as i64)
                            };
                            match cond {
                                0 => x == y,
                                1 => x != y,
                                2 => x < y,
                                3 => x <= y,
                                4 => x > y,
                                5 => x >= y,
                                _ => return Err("reserved integer compare".into()),
                            }
                        }
                    };
                    mask |= ((t && active(l)) as u32) << l;
                }
                return self.write_scalar(w, i.d, mask);
            }
            op::V_READLANE => {
                let lane = b[0] as usize & 15;
                return self.write_scalar(w, i.d, a[lane]);
            }
            op::V_READFIRSTLANE => {
                let lane = if exec == 0 { 0 } else { exec.trailing_zeros() as usize };
                return self.write_scalar(w, i.d, a[lane]);
            }
            op::V_WRITELANE => {
                w.v[i.d as usize][b[0] as usize & 15] = a[0];
                return Ok(());
            }
            _ => {}
        }
        let mut out = [0u32; 16];
        let mut carry = 0u32;
        for l in 0..16 {
            let (x, y, z) = (a[l], b[l], c[l]);
            let (fx, fy, fz) = (f(x), f(y), f(z));
            let bit = |m: u32| m >> l & 1;
            out[l] = match i.op {
                op::V_MOV => x,
                op::V_ADD => x.wrapping_add(y),
                op::V_SUB => x.wrapping_sub(y),
                op::V_MUL_LO => x.wrapping_mul(y),
                op::V_MUL_HI_U => ((x as u64 * y as u64) >> 32) as u32,
                op::V_MUL_HI_I => ((x as i32 as i64 * y as i32 as i64) >> 32) as u32,
                op::V_AND => x & y,
                op::V_OR => x | y,
                op::V_XOR => x ^ y,
                op::V_NOT => !x,
                op::V_SHL => x << (y & 31),
                op::V_SHR => x >> (y & 31),
                op::V_ASHR => ((x as i32) >> (y & 31)) as u32,
                op::V_BFE_U => bfe(x, y, false),
                op::V_BFE_I => bfe(x, y, true),
                op::V_BFI => (x & y) | (!x & z),
                op::V_MIN_I => (x as i32).min(y as i32) as u32,
                op::V_MIN_U => x.min(y),
                op::V_MAX_I => (x as i32).max(y as i32) as u32,
                op::V_MAX_U => x.max(y),
                op::V_ADD_CO => {
                    let (r, o) = x.overflowing_add(y);
                    carry |= ((o && active(l)) as u32) << l;
                    r
                }
                op::V_SUB_CO => {
                    let (r, o) = x.overflowing_sub(y);
                    carry |= ((o && active(l)) as u32) << l;
                    r
                }
                op::V_ADDC => x.wrapping_add(y).wrapping_add(bit(z)),
                op::V_SUBB => x.wrapping_sub(y).wrapping_sub(bit(z)),
                op::V_POPCNT => x.count_ones(),
                op::V_FFBL => if x == 0 { u32::MAX } else { x.trailing_zeros() },
                op::V_FFBH_U => if x == 0 { u32::MAX } else { 31 - x.leading_zeros() },
                op::V_FFBH_I => {
                    let m = if (x as i32) < 0 { !x } else { x };
                    if m == 0 { u32::MAX } else { 31 - m.leading_zeros() }
                }
                op::V_BFREV => x.reverse_bits(),
                op::V_PERM => {
                    let bytes = (x as u64) << 32 | y as u64;
                    (0..4).fold(0, |r, k| {
                        let sel = (z >> (8 * k)) & 0xff;
                        r | if sel < 8 { ((bytes >> (8 * sel)) & 0xff) as u32 } else { 0 } << (8 * k)
                    })
                }
                op::V_MORTON => (0..16).fold(0, |r, k| r | (x >> k & 1) << (2 * k) | (y >> k & 1) << (2 * k + 1)),
                op::V_CNDMASK => if bit(z) != 0 { y } else { x },
                op::V_ADD_F => canonical(fx + fy),
                op::V_SUB_F => canonical(fx - fy),
                op::V_MUL_F => canonical(fx * fy),
                op::V_FMA_F => canonical(fx.mul_add(fy, fz)),
                op::V_MIN_F => canonical(min_f(fx, fy)),
                op::V_MAX_F => canonical(max_f(fx, fy)),
                op::V_FLOOR => canonical(fx.floor()),
                op::V_CEIL => canonical(fx.ceil()),
                op::V_TRUNC => canonical(fx.trunc()),
                op::V_RNDNE => canonical(fx.round_ties_even()),
                op::V_FRACT => canonical((fx - fx.floor()).min(f32::from_bits(0x3f7f_ffff))),
                op::V_FREXP_MANT => {
                    if fx == 0.0 || !fx.is_finite() {
                        canonical(fx)
                    } else {
                        let e = (fx.abs().log2().floor() as i32) + 1;
                        let mut m = fx / 2f32.powi(e);
                        if m.abs() >= 1.0 {
                            m /= 2.0
                        } else if m.abs() < 0.5 {
                            m *= 2.0
                        }
                        m.to_bits()
                    }
                }
                op::V_FREXP_EXP => {
                    if fx == 0.0 || !fx.is_finite() {
                        0
                    } else {
                        let mut e = (fx.abs().log2().floor() as i32) + 1;
                        let m = fx.abs() / 2f32.powi(e);
                        if m >= 1.0 {
                            e += 1
                        } else if m < 0.5 {
                            e -= 1
                        }
                        e as u32
                    }
                }
                op::V_LDEXP => {
                    let e = (y as i32).clamp(-300, 300);
                    canonical((fx as f64 * 2f64.powi(e)) as f32)
                }
                op::V_RCP => canonical(1.0 / fx),
                op::V_RSQ => canonical(1.0 / fx.sqrt()),
                op::V_SQRT => canonical(fx.sqrt()),
                op::V_EXP2 => canonical(fx.exp2()),
                op::V_LOG2 => canonical(fx.log2()),
                op::V_SIN => canonical(((fx as f64) * std::f64::consts::TAU).sin() as f32),
                op::V_COS => canonical(((fx as f64) * std::f64::consts::TAU).cos() as f32),
                op::V_CVT_F_U => (x as f32).to_bits(),
                op::V_CVT_F_I => (x as i32 as f32).to_bits(),
                op::V_CVT_U_F => fx as u32,
                op::V_CVT_I_F => fx as i32 as u32,
                op::V_CVT_PK_F16 => f32_to_f16(fx) as u32 | (f32_to_f16(fy) as u32) << 16,
                op::V_CVT_F16_LO => canonical(f16_to_f32(x as u16)),
                op::V_CVT_F16_HI => canonical(f16_to_f32((x >> 16) as u16)),
                op::V_CUBEID => cube(fx, fy, fz).0.to_bits(),
                op::V_CUBESC => canonical(cube(fx, fy, fz).1),
                op::V_CUBETC => canonical(cube(fx, fy, fz).2),
                op::V_CUBEMA => canonical(cube(fx, fy, fz).3),
                op::V_PERMLANE => {
                    let s = y as usize & 15;
                    if active(s) { a[s] } else { 0 }
                }
                op::V_QUADPERM => {
                    let s = (l & !3) + ((y >> (2 * (l & 3))) & 3) as usize;
                    if active(s) { a[s] } else { 0 }
                }
                op::V_MBCNT => (x & ((1u32 << l) - 1)).count_ones(),
                op::V_INTERP | op::V_INTERP_FLAT => {
                    let prim = w.primitive[l] as usize;
                    let p = self.attributes.get(prim).ok_or("missing attribute block")?[cond as usize];
                    if i.op == op::V_INTERP_FLAT {
                        p[0].to_bits()
                    } else {
                        canonical(fy.mul_add(p[2], fx.mul_add(p[1], p[0])))
                    }
                }
                _ => return Err(format!("unimplemented vector op 0x{:02x}", i.op)),
            };
            if !lit && (i.hi >> 14) & 1 != 0 {
                let v = f(out[l]);
                out[l] = if v.is_nan() { 0 } else { v.clamp(0.0, 1.0).to_bits() };
            }
        }
        for l in 0..16 {
            if active(l) {
                w.v[i.d as usize][l] = out[l];
            }
        }
        if matches!(i.op, op::V_ADD_CO | op::V_SUB_CO) {
            self.write_scalar(w, i.hi as u8, carry)?;
        }
        Ok(())
    }

    fn write_scalar(&self, w: &mut Wave, code: u8, value: u32) -> Result<(), String> {
        match code {
            EXEC => w.exec = value as u16,
            SCALAR..=223 => w.s[(code - SCALAR) as usize] = value,
            _ => return Err("scalar destination".into()),
        }
        Ok(())
    }

    fn atomic(old: u64, operand: u64, compare: u64, operation: u32, wide: bool) -> u64 {
        let m = if wide { u64::MAX } else { u32::MAX as u64 };
        let sx = |v: u64| if wide { v as i64 } else { v as u32 as i32 as i64 };
        (match operation {
            0 => old.wrapping_add(operand),
            1 => operand,
            2 => if old & m == compare & m { operand } else { old },
            3 => old & operand,
            4 => old | operand,
            5 => old ^ operand,
            6 => if sx(operand) < sx(old) { operand } else { old },
            7 => if sx(operand) > sx(old) { operand } else { old },
            8 => (old & m).min(operand & m),
            _ => (old & m).max(operand & m),
        }) & m
    }

    fn memory_op(&mut self, w: &mut Wave, i: Inst) -> Result<(), String> {
        let offset = ((i.hi << 12) as i32 >> 12) as i64;
        let size = ((i.hi >> 20) & 3) as usize + 1;
        let operation = (i.hi >> 24) & 15;
        let returns = (i.hi >> 28) & 1 != 0;
        let spair = |w: &Wave, code: u8| {
            let r = (code - SCALAR) as usize;
            w.s[r] as u64 | (w.s[r + 1] as u64) << 32
        };
        match i.op {
            op::S_LOAD | op::S_BUFFER_LOAD => {
                let n = isa::scalar_load_dwords(i.hi >> 20) as usize;
                // The register offset and the immediate sum in 32 bits.
                let extra = if i.a == LITERAL { 0 } else { w.s[(i.a - SCALAR) as usize] };
                let byte = extra.wrapping_add(offset as u32) as u64;
                let d = (i.d - SCALAR) as usize;
                if i.op == op::S_LOAD {
                    let address = spair(w, i.b).wrapping_add(byte);
                    for k in 0..n {
                        w.s[d + k] = self.memory.read(address + 4 * k as u64)?;
                    }
                } else {
                    let desc = self.sgroup(w, i.b, 4);
                    let base = desc[0] as u64 | ((desc[1] as u64) << 32);
                    for k in 0..n {
                        let at = byte + 4 * k as u64;
                        w.s[d + k] = if at + 4 <= desc[2] as u64 { self.memory.read(base + at)? } else { 0 };
                    }
                }
                return Ok(());
            }
            _ => {}
        }
        let atomic = matches!(i.op, op::GLOBAL_ATOMIC | op::BUFFER_ATOMIC | op::SHARED_ATOMIC);
        let store = matches!(i.op, op::GLOBAL_STORE | op::BUFFER_STORE | op::SCRATCH_STORE | op::SHARED_STORE);
        let words = if atomic { size } else { size };
        let desc = if matches!(i.op, op::BUFFER_LOAD..=op::BUFFER_ATOMIC) { self.sgroup(w, i.b, 4) } else { vec![] };
        let words_private = self.program.private / 4;
        for l in 0..16 {
            if w.exec >> l & 1 == 0 {
                continue;
            }
            let va = |w: &Wave, k: u8| w.v[k as usize][l];
            // Per-lane dword addresses: Ok(Some(address)) or Ok(None) when dropped.
            enum Space {
                Global(u64),
                Buffer(i64),
                Shared(i64),
            }
            let space = match i.op {
                op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC => {
                    let base = if i.b == LITERAL {
                        if i.a == LITERAL { 0 } else { va(w, i.a) as u64 | (va(w, i.a + 1) as u64) << 32 }
                    } else {
                        let o = if i.a == LITERAL { 0 } else { va(w, i.a) };
                        spair(w, i.b).wrapping_add(o.wrapping_add(offset as u32) as u64)
                    };
                    Space::Global(if i.b == LITERAL { (base as i64 + offset) as u64 } else { base })
                }
                op::BUFFER_LOAD..=op::BUFFER_ATOMIC => {
                    Space::Buffer(if i.a == LITERAL { 0u32 } else { va(w, i.a) }.wrapping_add(offset as u32) as i64)
                }
                op::SCRATCH_LOAD | op::SCRATCH_STORE => {
                    let byte = if i.a == LITERAL { 0u32 } else { va(w, i.a) }.wrapping_add(offset as u32) as i64;
                    if byte % 4 != 0 || byte < 0 || byte / 4 + size as i64 > words_private as i64 {
                        return Err("private access out of bounds".into());
                    }
                    let word = byte as u64 / 4;
                    Space::Global(spair(w, i.b) + ((w.index as u64 * words_private as u64 + word) * 16 + l as u64) * 4)
                }
                _ => Space::Shared(
                    if i.a == LITERAL { 0 } else { va(w, i.a) }
                        .wrapping_add(if i.b == LITERAL { 0 } else { w.s[(i.b - SCALAR) as usize] })
                        .wrapping_add(offset as u32) as i64,
                ),
            };
            let scratch = matches!(i.op, op::SCRATCH_LOAD | op::SCRATCH_STORE);
            let address = |k: usize| -> Result<Option<(bool, u64)>, String> {
                Ok(match space {
                    Space::Global(a) => Some((false, a + if scratch { 64 * k as u64 } else { 4 * k as u64 })),
                    Space::Buffer(byte) => {
                        let byte = byte + 4 * k as i64;
                        if byte >= 0 && byte + 4 <= desc[2] as i64 {
                            Some((false, (desc[0] as u64 | (desc[1] as u64) << 32) + byte as u64))
                        } else if desc[3] & 1 != 0 {
                            None
                        } else {
                            return Err("buffer access out of range".into());
                        }
                    }
                    Space::Shared(byte) => {
                        let byte = byte + 4 * k as i64;
                        if byte < 0 || byte + 4 > self.program.shared as i64 || byte % 4 != 0 {
                            return Err("shared access out of range".into());
                        }
                        Some((true, byte as u64))
                    }
                })
            };
            let load = |m: &mut Self, at: (bool, u64)| -> Result<u32, String> {
                if at.0 {
                    let o = at.1 as usize;
                    Ok(u32::from_le_bytes(m.shared[o..o + 4].try_into().unwrap()))
                } else {
                    m.memory.read(at.1)
                }
            };
            let save = |m: &mut Self, at: (bool, u64), v: u32| -> Result<(), String> {
                if at.0 {
                    let o = at.1 as usize;
                    m.shared[o..o + 4].copy_from_slice(&v.to_le_bytes());
                    Ok(())
                } else {
                    m.memory.write(at.1, v)
                }
            };
            if atomic {
                let wide = size == 2;
                let (Some(lo), hi) = (address(0)?, if wide { address(1)? } else { None }) else {
                    if returns {
                        for k in 0..size {
                            w.v[i.d as usize + k][l] = 0;
                        }
                    }
                    continue;
                };
                let old = load(self, lo)? as u64 | if let Some(h) = hi { (load(self, h)? as u64) << 32 } else { 0 };
                let reg = |w: &Wave, k: usize| w.v[i.d as usize + k][l] as u64;
                let operand = reg(w, 0) | if wide { reg(w, 1) << 32 } else { 0 };
                let compare = if operation == 2 {
                    reg(w, size) | if wide { reg(w, size + 1) << 32 } else { 0 }
                } else {
                    0
                };
                let new = Self::atomic(old, operand, compare, operation, wide);
                save(self, lo, new as u32)?;
                if let Some(h) = hi {
                    save(self, h, (new >> 32) as u32)?;
                }
                if returns {
                    w.v[i.d as usize][l] = old as u32;
                    if wide {
                        w.v[i.d as usize + 1][l] = (old >> 32) as u32;
                    }
                }
                continue;
            }
            for k in 0..words {
                let at = address(k)?;
                if store {
                    if let Some(at) = at {
                        save(self, at, w.v[i.d as usize + k][l])?;
                    }
                } else {
                    w.v[i.d as usize + k][l] = match at {
                        Some(at) => load(self, at)?,
                        None => 0,
                    };
                }
            }
        }
        Ok(())
    }

    fn texel(&mut self, image: &[u32], level: u32, x: i64, y: i64, layer: i64) -> Result<[f32; 4], String> {
        let format = (image[1] >> 8) & 0xff;
        let width = ((image[2] & 0x3fff) + 1) as i64;
        let height = (((image[2] >> 14) & 0x3fff) + 1) as i64;
        let bpp = match format {
            4 => 16,
            _ => 4,
        };
        let mut base = image[0] as u64 | ((image[1] & 0xff) as u64) << 32;
        let pitch_units = image[5] as u64;
        let (mut w, mut h) = (width as u64, height as u64);
        let mut pitch = pitch_units * 64;
        for _ in 0..level {
            base += (pitch * h + 4095) & !4095;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            pitch = (w * bpp + 63) & !63;
        }
        let address = base + layer as u64 * image[6] as u64 * 64 + y as u64 * pitch + x as u64 * bpp;
        let word = |m: &mut Self, k: u64| m.memory.read(address + 4 * k);
        let unorm = |v: u32, s: u32| ((v >> s) & 0xff) as f32 / 255.0;
        Ok(match format {
            1 => {
                let v = word(self, 0)?;
                [unorm(v, 0), unorm(v, 8), unorm(v, 16), unorm(v, 24)]
            }
            2 => {
                let v = word(self, 0)?;
                [unorm(v, 16), unorm(v, 8), unorm(v, 0), unorm(v, 24)]
            }
            3 => [f32::from_bits(word(self, 0)?), 0.0, 0.0, 1.0],
            4 => [f(word(self, 0)?), f(word(self, 1)?), f(word(self, 2)?), f(word(self, 3)?)],
            5 => [f(word(self, 0)?), 0.0, 0.0, 1.0],
            _ => return Err(format!("texture format {format} outside the model")),
        })
    }

    fn texture(&mut self, w: &mut Wave, i: Inst) -> Result<(), String> {
        let image = self.sgroup(w, i.b, 8);
        let fetch = i.op == op::IMAGE_FETCH;
        let sampler = if fetch { vec![0; 8] } else { self.sgroup(w, i.hi as u8, 8) };
        let mask = (i.hi >> 8) & 15;
        let variant = (i.hi >> 14) & 7;
        let dim = (i.hi >> 19) & 7;
        if (i.hi >> 17) & 1 != 0 {
            return Err("depth compare outside the model".into());
        }
        let width = ((image[2] & 0x3fff) + 1) as f32;
        let height = (((image[2] >> 14) & 0x3fff) + 1) as f32;
        let levels = ((image[3] >> 11) & 15) + 1;
        let arrayed = matches!(dim, isa::DIM_1D_ARRAY | isa::DIM_2D_ARRAY);
        let coords2 = !matches!(dim, isa::DIM_1D | isa::DIM_1D_ARRAY);
        let n = isa::texture_coordinates(i.hi, fetch) as usize;
        let reg = |w: &Wave, k: usize, l: usize| w.v[i.a as usize + k][l];
        let base_count = match dim {
            isa::DIM_1D => 1,
            isa::DIM_2D => 2,
            isa::DIM_1D_ARRAY => 2,
            _ => 3,
        };
        let mut out = [[0f32; 4]; 16];
        for l in 0..16 {
            if w.exec >> l & 1 == 0 {
                continue;
            }
            let u = f(reg(w, 0, l));
            let v = if coords2 { f(reg(w, 1, l)) } else { 0.0 };
            let layer_index = if arrayed { base_count - 1 } else { usize::MAX };
            if fetch {
                let x = reg(w, 0, l) as i64;
                let y = if coords2 { reg(w, 1, l) as i64 } else { 0 };
                let layer = if arrayed { reg(w, layer_index, l) as i64 } else { 0 };
                let level = reg(w, n - 1, l);
                out[l] = self.texel(&image, level, x, y, layer)?;
                continue;
            }
            let layer = if arrayed { f(reg(w, layer_index, l)).round_ties_even().max(0.0) as i64 } else { 0 };
            let mut k = base_count;
            let mut lod = match variant {
                isa::TEX_LEVEL => {
                    k += 1;
                    f(reg(w, k - 1, l))
                }
                isa::TEX_SAMPLE | isa::TEX_BIAS | isa::TEX_GATHER => {
                    let bias = if variant == isa::TEX_BIAS {
                        k += 1;
                        f(reg(w, k - 1, l))
                    } else {
                        0.0
                    };
                    if variant == isa::TEX_GATHER || self.program.stage != Stage::Fragment {
                        bias
                    } else {
                        // Coarse quad derivatives: lane 1 − lane 0 and lane 2 − lane 0.
                        let q = l & !3;
                        let du = |m: usize| f(reg(w, 0, q + m)) - f(reg(w, 0, q));
                        let dv = |m: usize| if coords2 { f(reg(w, 1, q + m)) - f(reg(w, 1, q)) } else { 0.0 };
                        let (ux, vx) = (du(1) * width, dv(1) * height);
                        let (uy, vy) = (du(2) * width, dv(2) * height);
                        let rho2 = (ux * ux + vx * vx).max(uy * uy + vy * vy);
                        0.5 * rho2.log2() + bias
                    }
                }
                _ => return Err("texture variant outside the model".into()),
            };
            lod += ((sampler[1] & 0xffff) as i16) as f32 / 256.0;
            let max_lod = (sampler[2] & 0xfff) as f32 / 256.0;
            let min_lod = ((sampler[1] >> 16) & 0xfff) as f32 / 256.0;
            lod = lod.clamp(min_lod, max_lod.min((levels - 1) as f32));
            let magnify = lod <= 0.0;
            let linear = if magnify { sampler[0] & 1 != 0 } else { sampler[0] & 2 != 0 };
            let level = if magnify || (sampler[0] >> 2) & 3 == 0 { 0 } else { (lod + 0.5).floor() as u32 };
            let (lw, lh) = (((width as u32) >> level).max(1) as i64, ((height as u32) >> level).max(1) as i64);
            let wrap = |c: i64, size: i64, mode: u32| -> Option<i64> {
                match mode {
                    0 => Some(c.rem_euclid(size)),
                    1 => {
                        let m = c.rem_euclid(2 * size);
                        Some(if m >= size { 2 * size - 1 - m } else { m })
                    }
                    2 => Some(c.clamp(0, size - 1)),
                    3 => (0..size).contains(&c).then_some(c),
                    _ => Some((if c < 0 { -1 - c } else { c }).clamp(0, size - 1)),
                }
            };
            let (mode_u, mode_v) = ((sampler[0] >> 4) & 7, (sampler[0] >> 7) & 7);
            let (su, sv) = if sampler[0] >> 19 & 1 != 0 { (u, v) } else { (u * lw as f32, v * lh as f32) };
            let tap = |m: &mut Self, x: i64, y: i64| -> Result<[f32; 4], String> {
                match (wrap(x, lw, mode_u), if coords2 { wrap(y, lh, mode_v) } else { Some(0) }) {
                    (Some(x), Some(y)) => m.texel(&image, level, x, y, layer),
                    _ => Ok([0.0; 4]),
                }
            };
            out[l] = if variant == isa::TEX_GATHER {
                let (x0, y0) = ((su - 0.5).floor() as i64, (sv - 0.5).floor() as i64);
                let comp = ((i.hi >> 12) & 3) as usize;
                [tap(self, x0, y0 + 1)?[comp], tap(self, x0 + 1, y0 + 1)?[comp], tap(self, x0 + 1, y0)?[comp], tap(self, x0, y0)?[comp]]
            } else if linear {
                let (x, y) = (su - 0.5, if coords2 { sv - 0.5 } else { 0.0 });
                let (x0, y0) = (x.floor() as i64, y.floor() as i64);
                let (fx, fy) = (x - x.floor(), y - y.floor());
                let t00 = tap(self, x0, y0)?;
                let t10 = tap(self, x0 + 1, y0)?;
                let (t01, t11) = if coords2 { (tap(self, x0, y0 + 1)?, tap(self, x0 + 1, y0 + 1)?) } else { (t00, t10) };
                std::array::from_fn(|c| {
                    let top = t00[c] + (t10[c] - t00[c]) * fx;
                    let bottom = t01[c] + (t11[c] - t01[c]) * fx;
                    top + (bottom - top) * fy
                })
            } else {
                tap(self, su.floor() as i64, sv.floor() as i64)?
            };
        }
        for l in 0..16 {
            if w.exec >> l & 1 == 0 {
                continue;
            }
            let mut r = 0;
            for c in 0..4 {
                if mask >> c & 1 != 0 {
                    w.v[i.d as usize + r][l] = out[l][c].to_bits();
                    r += 1;
                }
            }
        }
        Ok(())
    }
}

impl Wave {
    fn new(exec: u16) -> Self {
        Self {
            v: vec![[0; 16]; 128],
            s: [0; 96],
            exec,
            launch: exec,
            pc: 0,
            done: false,
            barrier: false,
            index: 0,
            group: [0; 3],
            primitive: [0; 16],
        }
    }
}
