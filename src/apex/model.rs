// SPDX-License-Identifier: MIT
// Runs programs on the ISA reference model, `Tooling/apex_isa`, the
// executable form of Docs/isa.md that the RTL benches also use. A launch
// becomes a test directory of memory lines and WaveLaunch beats, which
// `python3 -m apex_isa run DIR --write` executes; its memory, export and
// fault files come back. `APEX_ISA` names the Tooling directory.
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::PathBuf;
use std::process::Command;

pub struct Region<'a> {
    pub gpuva: u64,
    pub data: &'a mut [u8],
}

/// Dwords of one attribute block in the ring (Docs/isa.md, Vertex outputs
/// and fragment inputs): two hidden rows and three rows per input.
pub const BLOCK: usize = 8 + 12 * 32;

/// One vertex or fragment wave. Fragment waves take the header's quad
/// origins from v2, coverage, primitive slot and facing from v3 (lane 4q for
/// per-quad fields), and primitive p's block at ring dword p · BLOCK.
pub struct Wave {
    pub exec: u16,
    pub scalar: [u32; 32],
    pub vector: [[u32; 16]; 72],
    pub attributes: Vec<[u32; BLOCK]>,
}
impl Default for Wave {
    fn default() -> Self {
        Self { exec: 0xffff, scalar: [0; 32], vector: [[0; 16]; 72], attributes: Vec::new() }
    }
}

pub enum Launch<'w> {
    /// A grid as the workgroup dispatcher launches it; wave k of the grid
    /// has private base `private_base + k · 16 · private bytes`.
    Compute { user: [u32; 16], groups: [u32; 3], private_base: u64 },
    Wave(&'w Wave),
}

/// Fragment exports by target: lane values and the lanes with coverage.
#[derive(Clone, Default)]
pub struct Exports {
    pub values: [[[u32; 4]; 16]; 10],
    pub lanes: [u16; 10],
}

const PROGRAM: u64 = 0xf000_0000;

/// A packed struct, first field in the most significant bits.
#[derive(Default)]
struct Bits(Vec<bool>);
impl Bits {
    fn push(&mut self, value: u64, width: usize) -> &mut Self {
        self.0.extend((0..width).rev().map(|b| b < 64 && (value >> b) & 1 != 0));
        self
    }
    /// An array of `logic<N, W>`: element N - 1 first.
    fn array(&mut self, values: &[u64], width: usize) -> &mut Self {
        for &v in values.iter().rev() {
            self.push(v, width);
        }
        self
    }
    fn hex(&self) -> String {
        let pad = (4 - self.0.len() % 4) % 4;
        let bits: Vec<bool> = std::iter::repeat_n(false, pad).chain(self.0.iter().copied()).collect();
        bits.chunks(4).map(|c| char::from_digit(c.iter().fold(0, |v, &b| v << 1 | b as u32), 16).unwrap()).collect()
    }
}

fn identity(h: &mut Bits, stage: u64, exec: u16, tag: u64) {
    h.push(stage, 2).push(1, 6).push(0, 6).push(0, 16).push(PROGRAM, 40).push(exec as u64, 16).push(tag, 8);
}

/// WaveLaunch beat: kind (header, register, attribute), last, target,
/// enable and 512 data bits.
fn beat(kind: u64, target: u64, enable: u16, data: &Bits) -> Bits {
    let mut b = Bits::default();
    b.push(kind, 2).push(0, 1).push(target, 8).push(enable as u64, 16);
    b.0.extend(&data.0);
    assert_eq!(b.0.len(), 539);
    b
}
fn lanes(values: &[u32; 16]) -> Bits {
    let mut b = Bits::default();
    b.array(&values.map(|v| v as u64), 32);
    b
}

fn wave_beats(stage: u64, w: &Wave) -> Vec<Bits> {
    let mut h = Bits::default();
    identity(&mut h, stage, w.exec, 0);
    let s = |k: usize| w.scalar[k] as u64;
    let mut skip = vec![];
    if stage == 1 {
        h.push(s(16) | s(17) << 32, 40).push(s(18), 32).push(s(19), 32).push(s(20), 32).push(0, 282);
    } else {
        let quad = |q: usize| (w.vector[2][4 * q] as u64, w.vector[3][4 * q] as u64);
        let coverage: Vec<u64> =
            (0..4).map(|q| (0..4).fold(0, |c, p| c | (w.vector[3][4 * q + p] as u64 & 0xff) << (8 * p))).collect();
        h.push(0, 4).push(0, 2).push(0, 6).push(s(16) & 31, 5).push(w.attributes.len().min(4) as u64, 3);
        h.array(&(0..4).map(|q| quad(q).0 & 0xffff).collect::<Vec<_>>(), 16);
        h.array(&(0..4).map(|q| quad(q).0 >> 16).collect::<Vec<_>>(), 16);
        h.array(&coverage, 32);
        h.array(&(0..4).map(|q| (quad(q).1 >> 8) & 3).collect::<Vec<_>>(), 2);
        h.array(&[(0..4).fold(0, |f, q| f | ((quad(q).1 >> 10) & 1) << q)], 4);
        h.array(&(0..4).map(|p| (p * BLOCK) as u64).collect::<Vec<_>>(), 12);
        h.push(0, 82);
        skip = vec![2, 3];
    }
    let mut out = vec![beat(0, 0, 0, &h)];
    out.push(beat(1, 128, 0xffff, &lanes(&w.scalar[..16].try_into().unwrap())));
    for (r, row) in w.vector.iter().enumerate() {
        if !skip.contains(&r) && (row.iter().any(|&v| v != 0) || r < 2) {
            out.push(beat(1, r as u64, 0xffff, &lanes(row)));
        }
    }
    let ring: Vec<u32> = w.attributes.iter().take(4).flatten().copied().collect();
    for (row, chunk) in ring.chunks(16).enumerate() {
        let mut v = [0; 16];
        v[..chunk.len()].copy_from_slice(chunk);
        out.push(beat(2, row as u64, 0xffff, &lanes(&v)));
    }
    out
}

fn compute_beats(p: &crate::isa::Program, user: &[u32; 16], groups: [u32; 3], private_base: u64) -> Vec<Vec<Bits>> {
    let invocations = p.invocations();
    let count = invocations.div_ceil(16);
    let mut waves = Vec::new();
    let mut k = 0u64;
    for z in 0..groups[2] {
        for y in 0..groups[1] {
            for x in 0..groups[0] {
                for wi in 0..count {
                    let lanes_left = invocations - 16 * wi;
                    let exec = if lanes_left >= 16 { 0xffff } else { (1u16 << lanes_left) - 1 };
                    let mut h = Bits::default();
                    identity(&mut h, 0, exec, k & 255);
                    h.push(x as u64, 32).push(y as u64, 32).push(z as u64, 32).push(wi as u64, 4).push(count as u64, 5);
                    h.push(p.shared.div_ceil(1024) as u64, 6);
                    h.push(p.local[0] as u64, 9).push(p.local[1] as u64, 9).push(p.local[2] as u64, 7);
                    h.push(private_base + k * 16 * p.private as u64, 40).push(0, 242);
                    waves.push(vec![beat(0, 0, 0, &h), beat(1, 128, 0xffff, &lanes(user))]);
                    k += 1;
                }
            }
        }
    }
    waves
}

fn tooling() -> Result<PathBuf, String> {
    std::env::var_os("APEX_ISA").map(PathBuf::from).ok_or_else(|| "APEX_ISA must name the Tooling directory".into())
}

/// Executes `program` (header and code) on the reference model with
/// `regions` as memory, which it updates; faults are errors.
pub fn run(program: &[u8], launch: &Launch, regions: &mut [Region]) -> Result<Exports, String> {
    let p = crate::isa::Program::parse(program)?;
    let waves = match launch {
        Launch::Compute { user, groups, private_base } => {
            if p.stage != crate::isa::Stage::Compute {
                return Err("compute launch of a graphics program".into());
            }
            compute_beats(&p, user, *groups, *private_base)
        }
        Launch::Wave(w) => {
            if p.stage == crate::isa::Stage::Compute {
                return Err("single-wave launch of a compute program".into());
            }
            vec![wave_beats(p.stage as u64, w)]
        }
    };
    let mut lines: BTreeMap<u64, [u8; 64]> = BTreeMap::new();
    let mut put = |va: u64, data: &[u8]| {
        for (i, &b) in data.iter().enumerate() {
            let a = va + i as u64;
            lines.entry(a & !63).or_insert([0; 64])[(a & 63) as usize] = b;
        }
    };
    put(PROGRAM, program);
    for r in regions.iter() {
        put(r.gpuva, r.data);
    }
    static SERIAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "apex-model-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut memory = String::new();
    for (a, l) in &lines {
        let _ = write!(memory, "{a:010x} ");
        for b in l.iter().rev() {
            let _ = write!(memory, "{b:02x}");
        }
        memory.push('\n');
    }
    let port = p.stage as u8;
    let mut launch_text = String::new();
    for mut beats in waves {
        let last = beats.last_mut().unwrap();
        last.0[2] = true;
        for b in beats {
            let _ = writeln!(launch_text, "{port} {}", b.hex());
        }
    }
    let files = [("memory.hex", memory), ("launch.hex", launch_text), ("config.txt", "abort_on_fault=1\n".into())];
    for (name, text) in files {
        std::fs::write(dir.join(name), text).map_err(|e| e.to_string())?;
    }
    let out = Command::new("python3")
        .args(["-m", "apex_isa", "run"])
        .arg(&dir)
        .arg("--write")
        .current_dir(tooling()?)
        .output()
        .map_err(|e| format!("python3: {e}"))?;
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
    let result = (|| {
        if !out.status.success() {
            return Err(format!("apex_isa: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        let faults = read("faults.txt");
        if !faults.trim().is_empty() {
            return Err(format!("fault (vmid queue status access address): {}", faults.trim()));
        }
        for line in read("expect.hex").lines() {
            let mut f = line.split_whitespace();
            let (Some(a), Some(d)) = (f.next(), f.next()) else { continue };
            let a = u64::from_str_radix(a, 16).map_err(|e| e.to_string())?;
            let bytes: Vec<u8> = (0..64).map(|i| u8::from_str_radix(&d[126 - 2 * i..128 - 2 * i], 16).unwrap()).collect();
            for r in regions.iter_mut() {
                let end = r.gpuva + r.data.len() as u64;
                for (i, &b) in bytes.iter().enumerate() {
                    let x = a + i as u64;
                    if x >= r.gpuva && x < end {
                        r.data[(x - r.gpuva) as usize] = b;
                    }
                }
            }
        }
        let mut e = Exports::default();
        for line in read("exports.txt").lines() {
            let f: Vec<u32> = line.split_whitespace().map(|x| u32::from_str_radix(x, 16).unwrap()).collect();
            let (quad, target, coverage) = (f[1] as usize, f[2] as usize, f[5]);
            for i in 0..4 {
                let l = 4 * quad + i;
                e.values[target][l].copy_from_slice(&f[6 + 4 * i..10 + 4 * i]);
                if (coverage >> (8 * i)) & 0xff != 0 {
                    e.lanes[target] |= 1 << l;
                }
            }
        }
        Ok(e)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}
