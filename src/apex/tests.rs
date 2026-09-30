// SPDX-License-Identifier: MIT
use crate::isa::{self, op, Access, Class, Inst, Kind, Program, Stage, LITERAL, SCALAR};
use crate::mir::{self, Op, Opnd, Value, CONST, COPY, LABEL};
use crate::sim::{Machine, Memory, Region, WaveInit};

/// A valid instance of every opcode, filling each field by its kind.
fn sample(opcode: u8) -> Inst {
    let hi = match isa::format(opcode).unwrap() {
        isa::Format::Memory if matches!(opcode, op::GLOBAL_ATOMIC | op::BUFFER_ATOMIC | op::SHARED_ATOMIC) => 2 << 24 | 1 << 28,
        isa::Format::Memory => 1 << 20 | 16,
        isa::Format::Texture if opcode == op::IMAGE_FETCH => 0xf << 8 | isa::TEX_FETCH << 14 | isa::DIM_2D << 19,
        isa::Format::Texture => 0xf << 8 | isa::DIM_2D << 19,
        _ => 0,
    };
    let kinds = isa::fields(opcode, hi).unwrap();
    let mut i = Inst { op: opcode, d: 0, a: 0, b: 0, hi };
    for (f, k) in kinds.iter().enumerate() {
        let code = match *k {
            Kind::None => continue,
            Kind::Raw => if opcode == op::EXP { 1 } else { 3 },
            Kind::S(_, n) => SCALAR + if n >= 4 { 8 } else { 2 * f as u8 },
            Kind::V(_, _) => 4 * f as u8,
            Kind::Any => [0, 5, SCALAR + 7, 241][f],
            Kind::SAny => [0, SCALAR + 3, 242, SCALAR + 5][f],
            Kind::OptS(_) => SCALAR + 10,
            Kind::OptV(_) => 20,
        };
        match f {
            0 => i.d = code,
            1 => i.a = code,
            2 => i.b = code,
            _ => i.hi = (i.hi & !0xff) | code as u32,
        }
    }
    i
}

#[test]
fn encoding() {
    for &(opcode, _) in isa::NAMES {
        let i = sample(opcode);
        i.validate().unwrap_or_else(|e| panic!("{}: {e}", isa::disassemble_one(i)));
        assert_eq!(Inst::decode(i.encode().unwrap()).unwrap(), i);
        let text = isa::disassemble_one(i);
        assert_eq!(isa::assemble_one(&text).unwrap_or_else(|e| panic!("{text}: {e}")), i, "{text}");
    }
    // Class bases and consecutive numbering.
    assert_eq!(op::S_LAUNCH, 0x45);
    assert_eq!(op::V_INTERP_FLAT, 0xab);
    assert_eq!(op::SHARED_ATOMIC, 0xcc);
    assert_eq!(op::EXP, 0xe2);
    for bad in [0x0b, 0x1f, 0x46, 0x5f, 0xac, 0xbf, 0xcd, 0xe3, 0xff] {
        assert!(Inst::decode(bad).is_err());
    }
    let add = Inst { op: op::V_ADD, d: 1, a: 2, b: 3, hi: 0 };
    assert!(add.validate().is_ok());
    for code in 226..240 {
        assert!(Inst { b: code, ..add }.validate().is_err(), "reserved operand {code}");
    }
    assert!(Inst { d: SCALAR, ..add }.validate().is_err());
    assert!(Inst { hi: 1 << 8, ..add }.validate().is_err(), "integer negate");
    assert!(Inst { hi: 1 << 15, ..add }.validate().is_err());
    let fadd = Inst { op: op::V_ADD_F, hi: 1 << 8 | 1 << 12 | 1 << 14, ..add };
    assert!(fadd.validate().is_ok());
    // A literal replaces c; three-source forms cannot carry one.
    assert!(Inst { op: op::V_FMA_F, a: LITERAL, hi: 0x3f80_0000, ..add }.validate().is_err());
    assert!(Inst { a: LITERAL, hi: 0x1234_5678, ..add }.validate().is_ok());
    assert!(Inst { op: op::V_CMP_U, d: SCALAR, b: LITERAL, hi: 7, ..add }.validate().is_err());
    // Pairs start at even registers; scalar loads write scalars.
    let s_add64 = Inst { op: op::S_ADD64, d: SCALAR + 2, a: SCALAR + 4, b: SCALAR + 6, hi: 0 };
    assert!(s_add64.validate().is_ok());
    assert!(Inst { a: SCALAR + 5, ..s_add64 }.validate().is_err());
    let load = Inst { op: op::S_LOAD, d: SCALAR + 4, a: LITERAL, b: SCALAR, hi: 2 << 20 };
    assert!(load.validate().is_ok());
    assert!(Inst { d: 4, ..load }.validate().is_err());
    assert!(Inst { hi: 1 << 30, ..load }.validate().is_err());
    let global = Inst { op: op::GLOBAL_LOAD, d: 0, a: 3, b: LITERAL, hi: 0 };
    assert!(global.validate().is_err(), "odd vector address pair");
    assert!(Inst { a: 4, ..global }.validate().is_ok());
    assert!(Inst { a: 3, b: SCALAR, ..global }.validate().is_ok(), "32-bit offset with a scalar base");
    assert!(Inst { op: op::EXP, d: 10, a: 0, b: 1, hi: 0 }.validate().is_err());
    assert!(Inst { op: op::S_NOP, hi: 1, ..Inst::new(0) }.validate().is_err());
}

#[test]
fn header() {
    let mut p = Program::new(Stage::Fragment, isa::assemble("v_interp v0, v0, v1, #4\nexp #0, v[0:3], #1\ns_endpgm").unwrap());
    p.flags = isa::EARLY_TESTS;
    p.output = 1;
    p.inputs[1] = 0x80 | 5;
    let bytes = p.bytes().unwrap();
    assert_eq!(bytes.len(), 64 + 24);
    assert_eq!(&bytes[..4], b"APXP");
    let q = Program::parse(&bytes).unwrap();
    assert_eq!((q.stage, q.flags, q.output, q.inputs[1]), (Stage::Fragment, 1, 1, 0x85));
    assert_eq!(Program::from_text(&q.text()).unwrap().bytes().unwrap(), bytes);
    let mut bad = bytes.clone();
    bad[0] ^= 1;
    assert!(Program::parse(&bad).is_err());
    assert!(Program::parse(&bytes[..bytes.len() - 8]).is_err());
    // Branches stay inside the program; graphics programs are one wave.
    let jump = Program::new(Stage::Compute, vec![Inst { op: op::S_BRANCH, hi: 5, ..Inst::new(0) }]);
    assert!(jump.bytes().is_err());
    let mut wide = Program::new(Stage::Vertex, vec![Inst::new(op::S_ENDPGM)]);
    wide.local = [32, 1, 1];
    assert!(wide.bytes().is_err());
    let mut compute = Program::new(Stage::Compute, vec![Inst::new(op::S_ENDPGM)]);
    compute.local = [256, 1, 1];
    assert!(compute.bytes().is_ok());
    compute.local = [256, 2, 1];
    assert!(compute.bytes().is_err());
}

/// Runs one compute wave of `text` with 16 lanes; returns the memory image.
fn run_compute(text: &str, exec: u32, memory: &mut [u8], private: u32) -> Result<(), String> {
    let mut p = Program::new(Stage::Compute, isa::assemble(text)?);
    p.local = [exec, 1, 1];
    p.private = private;
    p.shared = 256;
    p.validate()?;
    let mut scratch = vec![0u8; 16 * private as usize + 64];
    let mut m = Memory {
        regions: vec![Region { gpuva: 0x1000, data: memory }, Region { gpuva: 0x100000, data: &mut scratch }],
    };
    let mut machine = Machine::new(&p, &mut m);
    let user = [0x1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    machine.compute(&user, [1, 1, 1], 0x100000)
}
fn words(m: &[u8], at: usize, n: usize) -> Vec<u32> {
    (0..n).map(|k| u32::from_le_bytes(m[at + 4 * k..at + 4 * k + 4].try_into().unwrap())).collect()
}
/// Stores v`reg` of every lane at byte `at` of the root.
fn store(reg: u8, at: u32) -> String {
    format!("v_shl v100, lane, 2\nglobal_store v{reg}, v100, s[0:1] offset:{at} size:0\n")
}

#[test]
fn arithmetic() {
    let nan = f32::NAN.to_bits();
    let text = format!(
        "v_cvt_f_u v1, lane\n\
         v_min_f v2, v1, 0x{nan:08x}\n\
         v_max_f v3, -0.0, 0\n\
         v_mul_f v4, v1, 0.5 \n\
         v_fma_f v5, v1, v1, -v1\n\
         v_add_f v6, v1, -2.0 clamp\n\
         v_mov v7, 0x000000f0\n\
         v_bfe_i v7, v7, 0x0404\n\
         v_bfe_u v8, lane, 0x0201\n\
         v_ffbh_u v9, lane\n\
         v_ffbh_i v10, -2\n\
         v_mul_hi_u v11, -1, lane\n\
         v_cvt_u_f v12, -4.0\n\
         v_cvt_pk_f16 v13, v1, 0.5\n\
         v_cvt_f16_hi v14, v13\n\
         v_morton v15, lane, 1\n\
         v_rsq v16, 4.0\n\
         v_cmp_f s2, v1, 4.0, #2\n\
         v_cndmask v17, 0, 1, s2\n\
         v_mov v18, 0x0000ff00\n\
         v_mov v20, 0x12345678\n\
         v_bfi v18, v18, v20, 0\n\
         v_sub_f v19, 0x{nan:08x}, v1\n\
         {}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
        store(2, 0), store(3, 64), store(4, 128), store(5, 192), store(6, 256), store(7, 320),
        store(8, 384), store(9, 448), store(10, 512), store(11, 576), store(12, 640), store(13, 704),
        store(14, 768), store(15, 832), store(16, 896), store(17, 960), store(18, 1024), store(19, 1088), "s_endpgm"
    );
    let mut m = vec![0u8; 2048];
    run_compute(&text, 16, &mut m, 0).unwrap();
    for l in 0..16u32 {
        let f = |k: usize| f32::from_bits(words(&m, 64 * k, 16)[l as usize]);
        let w = |k: usize| words(&m, 64 * k, 16)[l as usize];
        assert_eq!(f(0), l as f32, "min returns the number");
        assert_eq!(w(1), 0, "max(-0, +0) = +0");
        assert_eq!(f(2), l as f32 * 0.5);
        assert_eq!(f(3), (l as f32).mul_add(l as f32, -(l as f32)));
        assert_eq!(f(4), (l as f32 - 2.0).clamp(0.0, 1.0));
        assert_eq!(w(5), 0xffff_ffff, "signed field 0xf");
        assert_eq!(w(6), (l >> 1) & 3);
        assert_eq!(w(7), if l == 0 { u32::MAX } else { 31 - l.leading_zeros() });
        assert_eq!(w(8), 0, "highest bit differing from the sign of -2");
        assert_eq!(w(9), if l == 0 { 0 } else { l - 1 });
        assert_eq!(w(10), 0, "saturating conversion");
        assert_eq!(w(11) >> 16, 0x3800, "0.5 in the high half");
        assert_eq!(f(12), 0.5);
        assert_eq!(w(13), (0..4).fold(0, |r, k| r | ((l >> k) & 1) << (2 * k)) | 2);
        assert_eq!(f(14), 0.5);
        assert_eq!(w(15), (l < 4) as u32);
        assert_eq!(w(16), 0x5600, "(mask & insert) | (~mask & base)");
        assert_eq!(w(17), 0x7fc0_0000, "canonical NaN");
    }
}

#[test]
fn exec_and_cross_lane() {
    let text = format!(
        "v_mov v99, 6\n\
         v_cmp_u s2, lane, v99, #2\n\
         s_mov s3, exec\n\
         s_and_saveexec s4, s2\n\
         v_mov v1, 7\n\
         v_mov v98, lane\n\
         v_readfirstlane s5, v98\n\
         s_andn2_saveexec s6, s4\n\
         v_mov v1, 9\n\
         s_mov exec, s3\n\
         v_add v2, lane, 100\n\
         v_readlane s7, v2, 13\n\
         v_mov v3, s7\n\
         v_permlane v4, v2, 15\n\
         v_quadperm v5, v2, 0x1b\n\
         v_mbcnt v6, 0x0000aaaa\n\
         s_mov s9, 3\n\
         v_writelane v7, 42, s9\n\
         v_mov v8, s5\n\
         v_mov v9, s6\n\
         {}{}{}{}{}{}{}{}{}s_endpgm",
        store(1, 0), store(3, 64), store(4, 128), store(5, 192), store(6, 256), store(7, 320),
        store(8, 384), store(9, 448), store(2, 512)
    );
    let mut m = vec![0u8; 1024];
    run_compute(&text, 16, &mut m, 0).unwrap();
    for l in 0..16u32 {
        let w = |k: usize| words(&m, 64 * k, 16)[l as usize];
        assert_eq!(w(0), if l < 6 { 7 } else { 9 });
        assert_eq!(w(1), 113);
        assert_eq!(w(2), 115);
        assert_eq!(w(3), 100 + (l & !3) + 3 - (l & 3), "pattern 0x1b reverses each quad");
        assert_eq!(w(4), (0xaaaa & ((1 << l) - 1) as u32).count_ones());
        assert_eq!(w(5), if l == 3 { 42 } else { 0 });
        assert_eq!(w(6), 0, "first active lane of the masked region");
        assert_eq!(w(7), 0x3f, "saveexec returns the old mask");
    }
    // Only launched lanes run; inactive source lanes read zero.
    let text = format!("v_add v1, lane, 1\nv_permlane v2, v1, 12\n{}s_endpgm", store(2, 0));
    let mut m = vec![0u8; 64];
    run_compute(&text, 10, &mut m, 0).unwrap();
    assert_eq!(words(&m, 0, 16), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn memory_model() {
    // Buffer bounds: robust descriptors read zero and drop stores; others fault.
    let text = "v_shl v1, lane, 2\n\
                s_load s[4:7], off, s[0:1] offset:512 size:2\n\
                buffer_load v2, v1, s[4:7] size:0\n\
                v_add v3, v2, 1\n\
                buffer_store v3, v1, s[4:7] offset:4 size:0\n\
                s_buffer_load s8, off, s[4:7] offset:60 size:0\n\
                s_buffer_load s9, off, s[4:7] offset:64 size:0\n\
                v_mov v4, s8\nv_mov v5, s9\n\
                global_store v4, v1, s[0:1] offset:256 size:0\n\
                global_store v5, v1, s[0:1] offset:320 size:0\n\
                s_endpgm";
    let mut m = vec![0u8; 1024];
    for k in 0..16u32 {
        m[4 * k as usize..4 * k as usize + 4].copy_from_slice(&(k * 10).to_le_bytes());
    }
    let desc = [0x1000u32, 0, 64, 1];
    for (k, d) in desc.iter().enumerate() {
        m[512 + 4 * k..516 + 4 * k].copy_from_slice(&d.to_le_bytes());
    }
    run_compute(text, 16, &mut m, 0).unwrap();
    let got = words(&m, 0, 16);
    for k in 1..16 {
        assert_eq!(got[k], (k as u32 - 1) * 10 + 1, "in-order load/store per wave");
    }
    assert_eq!(words(&m, 64, 1), [0], "store past the range dropped");
    assert_eq!(words(&m, 256, 1), [141], "after this wave's store to the same dword");
    assert_eq!(words(&m, 320, 1), [0], "scalar load past the range reads zero");
    m[512 + 12] = 0;
    assert!(run_compute(text, 16, &mut m, 0).unwrap_err().contains("out of range"));

    // Private memory interleaves lanes: base + ((wave * words + word) * 16 + lane) * 4.
    let text = "v_add v2, lane, 5\nv_add v3, lane, 6\nscratch_store v[2:3], off, s[20:21] offset:4 size:1\n\
                scratch_load v[4:5], off, s[20:21] offset:4 size:1\nv_add v6, v4, v5\n\
                v_shl v100, lane, 2\nglobal_store v6, v100, s[0:1] size:0\ns_endpgm";
    let mut m = vec![0u8; 64];
    run_compute(text, 16, &mut m, 12).unwrap();
    assert_eq!(words(&m, 0, 16), (0..16).map(|l| 2 * l + 11).collect::<Vec<_>>());
    assert!(run_compute(text, 16, &mut m, 8).unwrap_err().contains("private"));

    // Atomics at 32 and 64 bits; compare-exchange holds {new, compare}.
    let text = "v_mov v1, 1\nglobal_atomic v1, off, s[0:1] size:0 return:1\n\
                v_mov v2, 5\nv_mov v3, 16\nglobal_atomic v[2:3], off, s[0:1] atomic:2 return:1\n\
                v_mov v4, 0xffffffff\nv_mov v5, 0\nglobal_atomic v[4:5], off, s[0:1] offset:8 size:1\n\
                v_shl v100, lane, 2\nglobal_store v1, v100, s[0:1] offset:64 size:0\n\
                global_store v2, v100, s[0:1] offset:128 size:0\ns_endpgm";
    let mut m = vec![0u8; 256];
    m[8] = 1;
    run_compute(text, 16, &mut m, 0).unwrap();
    assert_eq!(words(&m, 0, 1), [5], "16 increments then one exchange from 16 to 5");
    assert_eq!(words(&m, 8, 2), [0xffff_fff1, 15], "1 + 16 * 0xffffffff carries into the high word");
    let old = words(&m, 64, 16);
    assert_eq!(old, (0..16).collect::<Vec<_>>());
    assert_eq!(words(&m, 128, 16).iter().filter(|&&v| v == 16).count(), 1);
}

#[test]
fn control_and_barriers() {
    // Four waves: each writes shared memory, meets at the barrier, reads a neighbour.
    let text = "s_launch s2, 3\nv_mov v1, s2\nv_shl v2, v0, 2\nshared_store v1, v2, off size:0\n\
                s_fence imm:13\ns_barrier\n\
                v_add v3, v0, 16\nv_and v3, v3, 63\nv_shl v3, v3, 2\nshared_load v4, v3, off size:0\n\
                s_cmp_eq s3, s2, 3\ns_cbranch_nz s3 offset:1\nv_add v4, v4, 100\n\
                global_store v4, v2, s[0:1] size:0\ns_endpgm";
    let mut p = Program::new(Stage::Compute, isa::assemble(text).unwrap());
    p.local = [64, 1, 1];
    p.shared = 256;
    let mut m = vec![0u8; 256];
    let mut mem = Memory { regions: vec![Region { gpuva: 0x1000, data: &mut m }] };
    Machine::new(&p, &mut mem).compute(&[0x1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1], 0).unwrap();
    let got = words(&m, 0, 64);
    for i in 0..64u32 {
        let neighbour = ((i + 16) % 64) / 16;
        assert_eq!(got[i as usize], neighbour + if i / 16 == 3 { 0 } else { 100 });
    }
    // execz skips a region whose mask is empty.
    let text = format!("v_cmp_u s2, lane, 0, #2\ns_and_saveexec s3, s2\ns_cbranch_execz offset:1\ns_trap imm:1\n\
                        s_mov exec, s3\nv_mov v1, 3\n{}s_endpgm", store(1, 0));
    let mut m = vec![0u8; 64];
    run_compute(&text, 16, &mut m, 0).unwrap();
    assert_eq!(words(&m, 0, 16), [3; 16]);
}

#[test]
fn fragment_interpolation_and_export() {
    let text = "v_interp v10, v0, v1, #6\nv_interp_flat v11, #7\nv_mov v12, 1.0\nv_mov v13, 0\n\
                exp #0, v[10:13], #1\ns_endpgm";
    let mut p = Program::new(Stage::Fragment, isa::assemble(text).unwrap());
    p.output = 1;
    p.inputs[1] = 0x80 | 3;
    let mut init = WaveInit { exec: 0x0ff0, scalar: [0; 32], vector: [[0; 16]; 72], primitive: [0; 16], attributes: vec![[[0.0; 3]; 144]; 2] };
    for l in 0..16 {
        init.vector[0][l] = (0.25 * (l & 3) as f32).to_bits();
        init.vector[1][l] = (0.5 * (l >> 2) as f32).to_bits();
        init.primitive[l] = (l >= 8) as u8;
    }
    init.attributes[0][6] = [1.0, 2.0, 4.0];
    init.attributes[1][6] = [-1.0, 8.0, 0.0];
    init.attributes[0][7] = [3.0, 9.0, 9.0];
    init.attributes[1][7] = [5.0, 9.0, 9.0];
    let mut none = Memory { regions: vec![] };
    let mut m = Machine::new(&p, &mut none);
    m.wave(&init).unwrap();
    assert_eq!(m.exports.lanes[0], 0x0ff0);
    for l in 4..12 {
        let (i, j) = (0.25 * (l & 3) as f32, 0.5 * (l >> 2) as f32);
        let a = init.attributes[(l >= 8) as usize][6];
        assert_eq!(f32::from_bits(m.exports.values[0][l][0]), j.mul_add(a[2], i.mul_add(a[1], a[0])));
        assert_eq!(f32::from_bits(m.exports.values[0][l][1]), if l >= 8 { 5.0 } else { 3.0 });
    }
    // A fragment program must end with a done export.
    let p = Program::new(Stage::Fragment, isa::assemble("s_endpgm").unwrap());
    let mut none = Memory { regions: vec![] };
    assert!(Machine::new(&p, &mut none).wave(&init).is_err());
}

fn v(id: u32) -> Opnd {
    Opnd::Val { id, off: 0, n: 1 }
}
fn phys(code: u8) -> Opnd {
    Opnd::Phys { code, n: 1 }
}
fn mop(op: u16, f: [Opnd; 4]) -> Op {
    Op::new(op, f)
}

/// `count` vector values live at once: each is lane + k, then all are summed.
fn pressure(count: u32, stage: Stage) -> Result<(Program, mir::Stats), String> {
    let mut values = vec![Value { class: Class::S, width: 1 }];
    let mut ops = Vec::new();
    for k in 0..count {
        values.push(Value { class: Class::V, width: 1 });
        let c = values.len() as u32;
        values.push(Value { class: Class::S, width: 1 });
        ops.push(Op { imm: k * 3 + 1000, ..mop(CONST, [v(c), Opnd::None, Opnd::None, Opnd::None]) });
        ops.push(mop(op::V_ADD as u16, [v(c - 1), phys(isa::LANE), v(c), Opnd::None]));
    }
    // A region boundary keeps the schedule from interleaving; the sum runs in
    // reverse definition order so every value stays live until its use.
    ops.push(Op { imm: 1, ..mop(LABEL, [Opnd::None; 4]) });
    let mut acc = 2 * count - 1;
    for k in (0..count - 1).rev() {
        values.push(Value { class: Class::V, width: 1 });
        let d = values.len() as u32 - 1;
        ops.push(mop(op::V_ADD as u16, [v(d), v(acc), v(2 * k + 1), Opnd::None]));
        acc = d;
    }
    values.push(Value { class: Class::S, width: 2 });
    let root = values.len() as u32 - 1;
    ops.push(mop(COPY, [Opnd::Val { id: root, off: 0, n: 1 }, phys(SCALAR), Opnd::None, Opnd::None]));
    ops.push(mop(COPY, [Opnd::Val { id: root, off: 1, n: 1 }, phys(SCALAR + 1), Opnd::None, Opnd::None]));
    values.push(Value { class: Class::V, width: 1 });
    let offset = values.len() as u32 - 1;
    values.push(Value { class: Class::S, width: 1 });
    ops.push(Op { imm: 2, ..mop(CONST, [v(offset + 1), Opnd::None, Opnd::None, Opnd::None]) });
    ops.push(mop(op::V_SHL as u16, [v(offset), phys(isa::LANE), v(offset + 1), Opnd::None]));
    ops.push(mop(op::GLOBAL_STORE as u16, [v(acc), v(offset), Opnd::Val { id: root, off: 0, n: 2 }, Opnd::None]));
    ops.push(mop(op::S_ENDPGM as u16, [Opnd::None; 4]));
    let mut header = Program::new(stage, vec![]);
    if stage == Stage::Fragment {
        header.output = 0;
    }
    mir::compile(ops, values, header)
}

#[test]
fn allocation_and_spilling() {
    let (p, stats) = pressure(126, Stage::Compute).unwrap();
    assert_eq!(stats.spills, 0);
    assert!(stats.vector <= 128);
    assert!(!p.code.iter().any(|i| i.op == op::S_NOP), "the scoreboard owns hazards");
    let run = |p: &Program| {
        let mut m = vec![0u8; 64];
        let mut scratch = vec![0u8; 16 * p.private as usize + 64];
        let mut mem = Memory { regions: vec![Region { gpuva: 0x1000, data: &mut m }, Region { gpuva: 0x100000, data: &mut scratch }] };
        Machine::new(p, &mut mem).compute(&[0x1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1], 0x100000).unwrap();
        words(&m, 0, 16)
    };
    let expect = |n: u32| (0..16).map(|l| (0..n).map(|k| l + k * 3 + 1000).sum::<u32>()).collect::<Vec<_>>();
    assert_eq!(run(&p), expect(126));
    // Beyond the register file a compute program spills and still computes.
    let (p, stats) = pressure(140, Stage::Compute).unwrap();
    assert!(stats.spills > 0 && p.private == 4 * stats.spills);
    assert_eq!(run(&p), expect(140));
    // Graphics launches have no private memory to spill into.
    assert!(pressure(140, Stage::Vertex).unwrap_err().contains("without spilling"));
}

#[test]
fn scalar_lane_spills() {
    // 110 uniform values live at once exceed 96 scalar registers; the excess
    // lives in vector lanes, including across a region with no active lane.
    let count = 110u32;
    let mut values = vec![Value { class: Class::S, width: 1 }];
    let mut ops = Vec::new();
    for k in 0..count {
        values.push(Value { class: Class::S, width: 1 });
        let d = values.len() as u32 - 1;
        ops.push(mop(op::S_ADD as u16, [v(d), phys(SCALAR + 16), Opnd::Lit(1000 + 7 * k), Opnd::None]));
    }
    values.push(Value { class: Class::S, width: 1 });
    let saved = values.len() as u32 - 1;
    ops.push(mop(COPY, [v(saved), phys(isa::EXEC), Opnd::None, Opnd::None]));
    ops.push(mop(COPY, [phys(isa::EXEC), Opnd::Lit(0), Opnd::None, Opnd::None]));
    ops.push(Op { imm: 1, ..mop(LABEL, [Opnd::None; 4]) });
    ops.push(mop(COPY, [phys(isa::EXEC), v(saved), Opnd::None, Opnd::None]));
    let mut acc = count;
    for k in (1..count).rev() {
        values.push(Value { class: Class::S, width: 1 });
        let d = values.len() as u32 - 1;
        ops.push(mop(op::S_ADD as u16, [v(d), v(acc), v(k), Opnd::None]));
        acc = d;
    }
    values.push(Value { class: Class::V, width: 1 });
    let x = values.len() as u32 - 1;
    values.push(Value { class: Class::S, width: 2 });
    let root = values.len() as u32 - 1;
    ops.push(mop(op::V_MOV as u16, [v(x), v(acc), Opnd::None, Opnd::None]));
    ops.push(mop(COPY, [Opnd::Val { id: root, off: 0, n: 1 }, phys(SCALAR), Opnd::None, Opnd::None]));
    ops.push(mop(COPY, [Opnd::Val { id: root, off: 1, n: 1 }, phys(SCALAR + 1), Opnd::None, Opnd::None]));
    ops.push(mop(op::GLOBAL_STORE as u16, [v(x), Opnd::None, Opnd::Val { id: root, off: 0, n: 2 }, Opnd::None]));
    for stage in [Stage::Compute, Stage::Vertex] {
        let (p, stats) = mir::compile(ops.clone(), values.clone(), Program::new(stage, vec![])).unwrap();
        assert!(stats.spills > 0 && stats.scalar <= 96 && p.private == 0);
        assert!(p.code.iter().any(|i| i.op == op::V_WRITELANE));
        if stage == Stage::Compute {
            let mut m = vec![0u8; 64];
            let mut mem = Memory { regions: vec![Region { gpuva: 0x1000, data: &mut m }] };
            let user = [0x1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            Machine::new(&p, &mut mem).compute(&user, [3, 1, 1], 0).unwrap();
            // The last workgroup (x = 2) writes last.
            assert_eq!(words(&m, 0, 1), [(0..count).map(|k| 2 + 1000 + 7 * k).sum::<u32>()]);
        }
    }
}

#[test]
fn divergent_loop_lifetimes() {
    // Lane l leaves the loop after l + 1 trips. x is defined in the loop and
    // read after it, so later loop values must not take its register: lanes
    // still running would overwrite x in their final trip.
    let mut values = vec![Value { class: Class::S, width: 1 }];
    let mut new = |class, width| {
        values.push(Value { class, width });
        values.len() as u32 - 1
    };
    use Class::{S, V};
    let (trip, x, junk, t, cmp, leaving, live, root, offset) =
        (new(V, 1), new(V, 1), new(V, 1), new(V, 1), new(S, 1), new(S, 1), new(S, 1), new(S, 2), new(V, 1));
    let e = phys(isa::EXEC);
    let pair = Opnd::Val { id: root, off: 0, n: 2 };
    let ops = vec![
        mop(COPY, [Opnd::Val { id: root, off: 0, n: 1 }, phys(SCALAR), Opnd::None, Opnd::None]),
        mop(COPY, [Opnd::Val { id: root, off: 1, n: 1 }, phys(SCALAR + 1), Opnd::None, Opnd::None]),
        mop(op::V_SHL as u16, [v(offset), phys(isa::LANE), Opnd::Lit(2), Opnd::None]),
        mop(COPY, [v(trip), Opnd::Lit(0), Opnd::None, Opnd::None]),
        mop(COPY, [v(live), e, Opnd::None, Opnd::None]),
        Op { imm: 1, ..mop(LABEL, [Opnd::None; 4]) },
        mop(COPY, [e, v(live), Opnd::None, Opnd::None]),
        mop(op::V_ADD as u16, [v(trip), v(trip), Opnd::Lit(1), Opnd::None]),
        mop(op::V_MUL_LO as u16, [v(x), v(trip), Opnd::Lit(10), Opnd::None]),
        mop(op::V_ADD as u16, [v(junk), v(trip), Opnd::Lit(77777), Opnd::None]),
        mop(op::V_ADD as u16, [v(t), v(junk), Opnd::Lit(1), Opnd::None]),
        Op { hi: 64, ..mop(op::GLOBAL_STORE as u16, [v(t), v(offset), pair, Opnd::None]) },
        mop(op::V_CMP_U as u16, [v(cmp), phys(isa::LANE), v(trip), Opnd::Raw(2)]),
        mop(op::S_AND as u16, [v(leaving), v(cmp), e, Opnd::None]),
        mop(op::S_ANDN2 as u16, [v(live), v(live), v(leaving), Opnd::None]),
        Op { imm: 1, ..mop(op::S_CBRANCH_NZ as u16, [Opnd::None, v(live), Opnd::None, Opnd::None]) },
        mop(COPY, [e, Opnd::Lit(0xffff), Opnd::None, Opnd::None]),
        mop(op::GLOBAL_STORE as u16, [v(x), v(offset), pair, Opnd::None]),
        mop(op::S_ENDPGM as u16, [Opnd::None; 4]),
    ];
    let (p, _) = mir::compile(ops, values, Program::new(Stage::Compute, vec![])).unwrap();
    let mut m = vec![0u8; 128];
    let mut mem = Memory { regions: vec![Region { gpuva: 0x1000, data: &mut m }] };
    Machine::new(&p, &mut mem).compute(&[0x1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1], 0).unwrap();
    assert_eq!(words(&m, 0, 16), (0..16).map(|l| 10 * (l + 1)).collect::<Vec<_>>());
    assert_eq!(words(&m, 64, 16), (0..16).map(|l| l + 1 + 77778).collect::<Vec<_>>());
}

#[test]
fn coalescing_and_constants() {
    // Values copied into a group become the group; small constants fold inline.
    let mut values = vec![Value { class: Class::S, width: 1 }];
    for w in [1, 1, 1, 1, 4, 2] {
        values.push(Value { class: if w == 2 { Class::S } else { Class::V }, width: w });
    }
    let g = 5;
    let mut ops = Vec::new();
    for k in 0..4u8 {
        ops.push(mop(op::V_ADD as u16, [v(1 + k as u32), phys(isa::LANE), Opnd::Lit(k as u32), Opnd::None]));
    }
    for k in 0..4u8 {
        ops.push(mop(COPY, [Opnd::Val { id: g, off: k, n: 1 }, v(1 + k as u32), Opnd::None, Opnd::None]));
    }
    ops.push(mop(COPY, [Opnd::Val { id: 6, off: 0, n: 1 }, phys(SCALAR), Opnd::None, Opnd::None]));
    ops.push(mop(COPY, [Opnd::Val { id: 6, off: 1, n: 1 }, phys(SCALAR + 1), Opnd::None, Opnd::None]));
    ops.push(Op { hi: 3 << 20, ..mop(op::GLOBAL_STORE as u16, [Opnd::Val { id: g, off: 0, n: 4 }, Opnd::None, Opnd::Val { id: 6, off: 0, n: 2 }, Opnd::None]) });
    let (p, stats) = mir::compile(ops, values, Program::new(Stage::Compute, vec![])).unwrap();
    assert_eq!(stats.instructions, 6, "{}", isa::disassemble(&p.code));
    assert!(p.code.iter().all(|i| i.op != op::V_MOV && i.op != op::S_MOV));
    let _ = Access::Use;
}
