// SPDX-License-Identifier: MIT
use crate::{isa, mir, schedule};
use std::ffi::{c_char, CStr};
#[repr(C)]
pub struct Input {
    op: u32,
    d: u32,
    a: u32,
    b: u32,
    c: u32,
    imm: u32,
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
pub unsafe extern "C" fn apex_emit(
    ptr: *const Input,
    count: usize,
    shared: u32,
    private: u32,
    path: *const c_char,
) -> i32 {
    report(
        std::panic::catch_unwind(|| {
            let input = if count == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(ptr, count) }
            };
            let ops = input
                .iter()
                .map(|i| {
                    Ok(mir::Op::new(
                        u8::try_from(i.op).map_err(|_| "opcode overflow")?,
                        i.d,
                        i.a,
                        i.b,
                        i.c,
                        i.imm,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let p = mir::compile(&ops, shared, private)?;
            eprintln!(
                "apex: {} instructions, s{} v{}, shared {}, private {}/lane",
                p.code.len(),
                p.scalar,
                p.vector,
                p.shared,
                p.private
            );
            let path = unsafe { CStr::from_ptr(path) }
                .to_str()
                .map_err(|_| "path UTF8")?;
            std::fs::write(path, p.bytes()?).map_err(|e| e.to_string())
        })
        .unwrap_or_else(|_| Err("compiler panic".into())),
    )
}
#[no_mangle]
pub unsafe extern "C" fn apex_tool(
    mode: *const c_char,
    input: *const c_char,
    output: *const c_char,
) -> i32 {
    report(
        std::panic::catch_unwind(|| {
            let mode = unsafe { CStr::from_ptr(mode) }
                .to_str()
                .map_err(|_| "mode UTF8")?;
            let input = unsafe { CStr::from_ptr(input) }
                .to_str()
                .map_err(|_| "path UTF8")?;
            let output = unsafe { CStr::from_ptr(output) }
                .to_str()
                .map_err(|_| "path UTF8")?;
            let data = std::fs::read(input).map_err(|e| e.to_string())?;
            if mode == "--disassemble" || mode == "--validate" {
                let p = isa::Program::parse(&data)?;
                schedule::validate(&p.code)?;
                if mode == "--disassemble" {
                    let text = format!(
                        "{} {} {} {} {}\n{}",
                        p.entry,
                        p.scalar,
                        p.vector,
                        p.shared,
                        p.private,
                        isa::disassemble(&p.code)
                    );
                    std::fs::write(output, text).map_err(|e| e.to_string())?;
                }
            } else if mode == "--mir" {
                let text = std::str::from_utf8(&data).map_err(|_| "MIR UTF8")?;
                let (head, body) = text.split_once('\n').ok_or("missing shared bytes")?;
                let shared = head.parse().map_err(|_| "invalid shared bytes")?;
                let mut ops = Vec::new();
                for line in body.lines().filter(|l| !l.trim().is_empty()) {
                    let fields: Vec<_> = line.split_whitespace().collect();
                    if fields.len() != 6 {
                        return Err("MIR: mnemonic dst src0 src1 src2 immediate".into());
                    }
                    let op = match fields[0] {
                        "buffer_address" => 0xf0,
                        "pair" => 0xf1,
                        "scalar_word" => 0xf2,
                        name => {
                            isa::NAMES
                                .iter()
                                .find(|(_, n)| *n == name)
                                .ok_or("unknown MIR operation")?
                                .0
                        }
                    };
                    let p = fields[1..]
                        .iter()
                        .map(|s| s.parse::<u32>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|_| "invalid MIR value")?;
                    ops.push(mir::Op::new(op, p[0], p[1], p[2], p[3], p[4]));
                }
                let p = mir::compile(&ops, shared, 0)?;
                std::fs::write(output, p.bytes()?).map_err(|e| e.to_string())?;
            } else if mode == "--assemble" {
                // Assembly preserves all variable launch requirements.
                let text = std::str::from_utf8(&data).map_err(|_| "assembly UTF8")?;
                let (head, body) = text.split_once('\n').ok_or("missing metadata")?;
                let h: Vec<u32> = head
                    .split_whitespace()
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| "bad metadata")?;
                if h.len() != 5 {
                    return Err("metadata: entry scalar vector shared private".into());
                }
                let p = isa::Program {
                    code: isa::assemble(body)?,
                    entry: h[0],
                    scalar: h[1],
                    vector: h[2],
                    shared: h[3],
                    private: h[4],
                };
                schedule::validate(&p.code)?;
                std::fs::write(output, p.bytes()?).map_err(|e| e.to_string())?;
            } else {
                return Err("unknown tool mode".into());
            }
            Ok(())
        })
        .unwrap_or_else(|_| Err("tool panic".into())),
    )
}
