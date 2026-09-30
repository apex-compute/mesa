// SPDX-License-Identifier: MIT
use crate::isa::{self, Class, Program, Stage};
use crate::mir::{self, Opnd};
use crate::sim;
use std::ffi::{c_char, CStr};

#[repr(C)]
pub struct Input {
    op: u32,
    f: [u32; 4],
    hi: u32,
    imm: u32,
    flags: u32,
}
#[repr(C)]
pub struct ValueDecl {
    class: u8,
    width: u8,
}
#[repr(C)]
pub struct Header {
    stage: u32,
    flags: u32,
    local: [u32; 3],
    shared: u32,
    private_bytes: u32,
    output: u32,
    inputs: [u8; 32],
}
#[repr(C)]
pub struct CompileResult {
    data: *mut u8,
    size: usize,
    instructions: u32,
    vector: u32,
    scalar: u32,
    spills: u32,
    diagnostic: [u8; 1024],
}
impl Default for CompileResult {
    fn default() -> Self {
        Self { data: std::ptr::null_mut(), size: 0, instructions: 0, vector: 0, scalar: 0, spills: 0, diagnostic: [0; 1024] }
    }
}
#[no_mangle]
pub unsafe extern "C" fn apex_compile_result_finish(result: &mut CompileResult) {
    if !result.data.is_null() {
        unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(result.data, result.size))) };
    }
    *result = CompileResult::default();
}

fn operand(x: u32) -> Opnd {
    if x == 0 {
        Opnd::None
    } else if x >> 31 != 0 {
        Opnd::Val { id: x & 0xff_ffff, off: ((x >> 24) & 7) as u8, n: ((x >> 27) & 7) as u8 + 1 }
    } else if x >> 30 != 0 {
        Opnd::Phys { code: x as u8, n: ((x >> 27) & 7) as u8 + 1 }
    } else {
        Opnd::Raw(x as u8)
    }
}
fn stage(s: u32) -> Result<Stage, String> {
    Ok(match s {
        0 => Stage::Compute,
        1 => Stage::Vertex,
        2 => Stage::Fragment,
        _ => return Err("invalid stage".into()),
    })
}

/// Compiles machine IR with its value table into a program.
#[no_mangle]
pub unsafe extern "C" fn apex_emit(
    ptr: *const Input,
    count: usize,
    decls: *const ValueDecl,
    value_count: usize,
    header: &Header,
    result: &mut CompileResult,
) -> i32 {
    *result = CompileResult::default();
    let compiled = std::panic::catch_unwind(|| -> Result<(Vec<u8>, mir::Stats), String> {
        let input = if count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, count) } };
        let decls = if value_count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(decls, value_count) } };
        let values = decls
            .iter()
            .map(|d| mir::Value { class: if d.class != 0 { Class::V } else { Class::S }, width: d.width })
            .collect();
        let ops = input
            .iter()
            .map(|i| {
                if i.op > 0x1ff {
                    return Err("opcode overflow".to_string());
                }
                Ok(mir::Op { op: i.op as u16, f: i.f.map(operand), hi: i.hi, imm: i.imm, flags: i.flags })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut p = Program::new(stage(header.stage)?, Vec::new());
        p.flags = header.flags as u8;
        p.local = header.local;
        p.shared = header.shared;
        p.private = header.private_bytes;
        p.output = header.output;
        p.inputs = header.inputs;
        let (p, stats) = mir::compile(ops, values, p)?;
        Ok((p.bytes()?, stats))
    })
    .unwrap_or_else(|_| Err("compiler panic".into()));
    match compiled {
        Ok((bytes, stats)) => {
            let bytes = Box::leak(bytes.into_boxed_slice());
            result.data = bytes.as_mut_ptr();
            result.size = bytes.len();
            result.instructions = stats.instructions;
            result.vector = stats.vector;
            result.scalar = stats.scalar;
            result.spills = stats.spills;
            0
        }
        Err(error) => {
            let count = error.len().min(result.diagnostic.len() - 1);
            result.diagnostic[..count].copy_from_slice(&error.as_bytes()[..count]);
            1
        }
    }
}

#[repr(C)]
pub struct SimRegion {
    gpuva: u64,
    data: *mut u8,
    size: u64,
}
#[repr(C)]
pub struct SimWave {
    exec: u32,
    scalar: [u32; 32],
    vector: [[u32; 16]; 72],
    primitive: [u8; 16],
    /// `primitives` blocks of 144 components × {P0, P10, P20}.
    attributes: *const f32,
    primitives: u32,
    exports: [[[u32; 4]; 16]; 10],
    exported: [u32; 10],
}

/// Test model: runs a compute grid, or one vertex/fragment wave when `wave` is set.
#[no_mangle]
pub unsafe extern "C" fn apex_simulate(
    program: *const u8,
    size: usize,
    user: &[u32; 16],
    groups: &[u32; 3],
    private_base: u64,
    wave: *mut SimWave,
    regions: *mut SimRegion,
    count: usize,
    executed: &mut u64,
    diagnostic: *mut c_char,
) -> i32 {
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<u64, String> {
        let bytes = unsafe { std::slice::from_raw_parts(program, size) };
        let p = Program::parse(bytes)?;
        let raw = if count == 0 { &mut [][..] } else { unsafe { std::slice::from_raw_parts_mut(regions, count) } };
        let mut memory = sim::Memory {
            regions: raw
                .iter_mut()
                .map(|r| sim::Region { gpuva: r.gpuva, data: unsafe { std::slice::from_raw_parts_mut(r.data, r.size as usize) } })
                .collect(),
        };
        let mut m = sim::Machine::new(&p, &mut memory);
        if let Some(w) = unsafe { wave.as_mut() } {
            let blocks = if w.primitives == 0 {
                Vec::new()
            } else {
                let a = unsafe { std::slice::from_raw_parts(w.attributes, w.primitives as usize * 144 * 3) };
                a.chunks_exact(432).map(|c| std::array::from_fn(|k| [c[3 * k], c[3 * k + 1], c[3 * k + 2]])).collect()
            };
            let init = sim::WaveInit { exec: w.exec as u16, scalar: w.scalar, vector: w.vector, primitive: w.primitive, attributes: blocks };
            m.wave(&init)?;
            w.exports = m.exports.values;
            w.exported = m.exports.lanes.map(|x| x as u32);
        } else {
            m.compute(user, *groups, private_base)?;
        }
        Ok(m.stats.instructions)
    }))
    .unwrap_or_else(|_| Err("simulator panic".into()));
    match run {
        Ok(n) => {
            *executed = n;
            0
        }
        Err(e) => {
            if !diagnostic.is_null() {
                let b = e.as_bytes();
                let n = b.len().min(255);
                unsafe {
                    std::ptr::copy_nonoverlapping(b.as_ptr(), diagnostic as *mut u8, n);
                    *diagnostic.add(n) = 0;
                }
            }
            1
        }
    }
}

fn report(result: Result<(), String>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("apex: {e}");
            1
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn apex_tool(mode: *const c_char, input: *const c_char, output: *const c_char) -> i32 {
    report(
        std::panic::catch_unwind(|| {
            let mode = unsafe { CStr::from_ptr(mode) }.to_str().map_err(|_| "mode UTF8")?;
            let input = unsafe { CStr::from_ptr(input) }.to_str().map_err(|_| "path UTF8")?;
            let output = unsafe { CStr::from_ptr(output) }.to_str().map_err(|_| "path UTF8")?;
            let data = std::fs::read(input).map_err(|e| e.to_string())?;
            match mode {
                "--disassemble" | "--validate" => {
                    let p = isa::Program::parse(&data)?;
                    if mode == "--disassemble" {
                        std::fs::write(output, p.text()).map_err(|e| e.to_string())?;
                    }
                }
                "--assemble" => {
                    let text = std::str::from_utf8(&data).map_err(|_| "assembly UTF8")?;
                    let p = isa::Program::from_text(text)?;
                    std::fs::write(output, p.bytes()?).map_err(|e| e.to_string())?;
                }
                _ => return Err("unknown tool mode".into()),
            }
            Ok(())
        })
        .unwrap_or_else(|_| Err("tool panic".into())),
    )
}
