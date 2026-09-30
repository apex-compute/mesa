// SPDX-License-Identifier: MIT
// P7 shader-core ISA: one little-endian 64-bit word per instruction.
// lo[7:0] opcode, lo[15:8] d, lo[23:16] a, lo[31:24] b, hi format-specific.

macro_rules! opcodes {
    ($($name:ident = $value:expr),* $(,)?) => {
        pub mod op {
            $(pub const $name: u8 = $value;)*
        }
        pub const NAMES: &[(u8, &str)] = &[$((op::$name, stringify!($name))),*];
    };
}

// Opcodes are consecutive from their class base in the order of the contract.
opcodes! {
    S_NOP = 0x00, S_ENDPGM = 0x01, S_TRAP = 0x02, S_BRANCH = 0x03, S_CBRANCH_Z = 0x04,
    S_CBRANCH_NZ = 0x05, S_CBRANCH_EXECZ = 0x06, S_CBRANCH_EXECNZ = 0x07, S_BARRIER = 0x08,
    S_FENCE = 0x09, S_SLEEP = 0x0a,

    S_MOV = 0x20, S_ADD = 0x21, S_SUB = 0x22, S_MUL = 0x23, S_MUL_HI_U = 0x24, S_AND = 0x25,
    S_OR = 0x26, S_XOR = 0x27, S_ANDN2 = 0x28, S_ORN2 = 0x29, S_NOT = 0x2a, S_SHL = 0x2b,
    S_SHR = 0x2c, S_ASHR = 0x2d, S_BFE_U = 0x2e, S_BFE_I = 0x2f, S_MIN_I = 0x30, S_MIN_U = 0x31,
    S_MAX_I = 0x32, S_MAX_U = 0x33, S_CMP_EQ = 0x34, S_CMP_NE = 0x35, S_CMP_LT_I = 0x36,
    S_CMP_LT_U = 0x37, S_CMP_LE_I = 0x38, S_CMP_LE_U = 0x39, S_CSELECT = 0x3a, S_ADD64 = 0x3b,
    S_SUB64 = 0x3c, S_SHL64 = 0x3d, S_FF1 = 0x3e, S_POPCNT = 0x3f, S_AND_SAVEEXEC = 0x40,
    S_OR_SAVEEXEC = 0x41, S_ANDN2_SAVEEXEC = 0x42, S_SETEXEC = 0x43, S_MEMTIME = 0x44,
    S_LAUNCH = 0x45,

    V_MOV = 0x60, V_ADD = 0x61, V_SUB = 0x62, V_MUL_LO = 0x63, V_MUL_HI_U = 0x64,
    V_MUL_HI_I = 0x65, V_AND = 0x66, V_OR = 0x67, V_XOR = 0x68, V_NOT = 0x69, V_SHL = 0x6a,
    V_SHR = 0x6b, V_ASHR = 0x6c, V_BFE_U = 0x6d, V_BFE_I = 0x6e, V_BFI = 0x6f, V_MIN_I = 0x70,
    V_MIN_U = 0x71, V_MAX_I = 0x72, V_MAX_U = 0x73, V_ADD_CO = 0x74, V_SUB_CO = 0x75,
    V_ADDC = 0x76, V_SUBB = 0x77, V_POPCNT = 0x78, V_FFBL = 0x79, V_FFBH_U = 0x7a,
    V_FFBH_I = 0x7b, V_BFREV = 0x7c, V_PERM = 0x7d, V_MORTON = 0x7e, V_CNDMASK = 0x7f,
    V_CMP_I = 0x80, V_CMP_U = 0x81, V_CMP_F = 0x82, V_CMP_CLASS = 0x83, V_ADD_F = 0x84,
    V_SUB_F = 0x85, V_MUL_F = 0x86, V_FMA_F = 0x87, V_MIN_F = 0x88, V_MAX_F = 0x89,
    V_FLOOR = 0x8a, V_CEIL = 0x8b, V_TRUNC = 0x8c, V_RNDNE = 0x8d, V_FRACT = 0x8e,
    V_FREXP_MANT = 0x8f, V_FREXP_EXP = 0x90, V_LDEXP = 0x91, V_RCP = 0x92, V_RSQ = 0x93,
    V_SQRT = 0x94, V_EXP2 = 0x95, V_LOG2 = 0x96, V_SIN = 0x97, V_COS = 0x98, V_CVT_F_U = 0x99,
    V_CVT_F_I = 0x9a, V_CVT_U_F = 0x9b, V_CVT_I_F = 0x9c, V_CVT_PK_F16 = 0x9d,
    V_CVT_F16_LO = 0x9e, V_CVT_F16_HI = 0x9f, V_CUBEID = 0xa0, V_CUBESC = 0xa1,
    V_CUBETC = 0xa2, V_CUBEMA = 0xa3, V_READLANE = 0xa4, V_READFIRSTLANE = 0xa5,
    V_WRITELANE = 0xa6, V_PERMLANE = 0xa7, V_QUADPERM = 0xa8, V_MBCNT = 0xa9, V_INTERP = 0xaa,
    V_INTERP_FLAT = 0xab,

    S_LOAD = 0xc0, S_BUFFER_LOAD = 0xc1, GLOBAL_LOAD = 0xc2, GLOBAL_STORE = 0xc3,
    GLOBAL_ATOMIC = 0xc4, BUFFER_LOAD = 0xc5, BUFFER_STORE = 0xc6, BUFFER_ATOMIC = 0xc7,
    SCRATCH_LOAD = 0xc8, SCRATCH_STORE = 0xc9, SHARED_LOAD = 0xca, SHARED_STORE = 0xcb,
    SHARED_ATOMIC = 0xcc,

    IMAGE_SAMPLE = 0xe0, IMAGE_FETCH = 0xe1, EXP = 0xe2,
}

pub const VECTOR_REGISTERS: u8 = 128;
pub const SCALAR_REGISTERS: u8 = 96;
pub const SCALAR: u8 = 128;
pub const EXEC: u8 = 224;
pub const LANE: u8 = 225;
pub const LITERAL: u8 = 255;
/// Operand codes 240-254: integers, then binary32 floats.
pub const INLINE: [u32; 15] = [
    0, 1, 2, 4, 8, 0xffff_ffff, 0xffff_fffe, 0xffff_fffc, 0x3f00_0000, 0x3f80_0000,
    0x4000_0000, 0x4080_0000, 0xbf00_0000, 0xbf80_0000, 0xc000_0000,
];
pub fn inline_code(value: u32) -> Option<u8> {
    INLINE.iter().position(|&v| v == value).map(|i| 240 + i as u8)
}

// Texture hi fields.
pub const TEX_SAMPLE: u32 = 0;
pub const TEX_LEVEL: u32 = 1;
pub const TEX_BIAS: u32 = 2;
pub const TEX_GRADIENT: u32 = 3;
pub const TEX_FETCH: u32 = 4;
pub const TEX_GATHER: u32 = 5;
pub const DIM_1D: u32 = 0;
pub const DIM_2D: u32 = 1;
pub const DIM_3D: u32 = 2;
pub const DIM_CUBE: u32 = 3;
pub const DIM_1D_ARRAY: u32 = 4;
pub const DIM_2D_ARRAY: u32 = 5;
pub const DIM_CUBE_ARRAY: u32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Control,
    Salu,
    Valu,
    Memory,
    Texture,
    Export,
}
pub fn format(op: u8) -> Option<Format> {
    Some(match op {
        0x00..=0x0a => Format::Control,
        0x20..=0x45 => Format::Salu,
        0x60..=0xab => Format::Valu,
        0xc0..=0xcc => Format::Memory,
        0xe0 | 0xe1 => Format::Texture,
        0xe2 => Format::Export,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    S,
    V,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Def,
    Use,
    DefUse,
}
/// What an instruction field may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    None,
    /// A raw 8-bit number (compare condition, attribute component, export target/flags).
    Raw,
    /// Scalar registers only (scalar ALU/branch sources also take exec and constants).
    S(Access, u8),
    /// Vector registers only.
    V(Access, u8),
    /// Vector-ALU source: vector or scalar register, exec, lane index or constant.
    Any,
    /// Scalar-ALU source: scalar register, exec or constant.
    SAny,
    /// Optional register: `255` means absent.
    OptS(u8),
    OptV(u8),
}

/// Sources taking the FP modifiers (bit 0 a, 1 b, 2 c).
pub fn fp_sources(op: u8) -> u32 {
    match op {
        op::V_FMA_F | op::V_CUBEID..=op::V_CUBEMA => 7,
        op::V_ADD_F..=op::V_MUL_F | op::V_MIN_F | op::V_MAX_F | op::V_CMP_F | op::V_CVT_PK_F16 => 3,
        op::V_FLOOR..=op::V_LDEXP | op::V_RCP..=op::V_COS | op::V_CVT_U_F | op::V_CVT_I_F | op::V_CMP_CLASS => 1,
        _ => 0,
    }
}
/// Operations with an FP32 result, which may clamp.
pub fn fp_result(op: u8) -> bool {
    matches!(op, op::V_ADD_F..=op::V_FREXP_MANT | op::V_LDEXP..=op::V_CVT_F_I
        | op::V_CVT_F16_LO..=op::V_CUBEMA)
}

pub fn scalar_load_dwords(code: u32) -> u8 {
    [1, 2, 4, 8][code as usize & 3]
}
/// Coordinate registers: four, or eight for gradients, compare with a level or
/// bias, and offsets.
pub fn texture_coordinates(hi: u32) -> u8 {
    let variant = (hi >> 14) & 7;
    let compare = (hi >> 17) & 1 != 0;
    if variant == TEX_GRADIENT || (compare && matches!(variant, TEX_LEVEL | TEX_BIAS)) || (hi >> 18) & 1 != 0 {
        8
    } else {
        4
    }
}

/// Field kinds of d, a, b and c (c is `hi[7:0]` in the ALU and texture formats).
pub fn fields(op: u8, hi: u32) -> Result<[Kind; 4], String> {
    use Access::*;
    use Kind::*;
    let none = [None; 4];
    let size = ((hi >> 20) & 3) as u8 + 1;
    Ok(match op {
        op::S_NOP..=op::S_BRANCH | op::S_CBRANCH_EXECZ..=op::S_SLEEP => none,
        op::S_CBRANCH_Z | op::S_CBRANCH_NZ => [None, SAny, None, None],
        op::S_MOV | op::S_NOT | op::S_FF1 | op::S_POPCNT | op::S_AND_SAVEEXEC..=op::S_ANDN2_SAVEEXEC => {
            [S(Def, 1), SAny, None, None]
        }
        op::S_SETEXEC => [None, SAny, None, None],
        op::S_CSELECT => [S(Def, 1), SAny, SAny, SAny],
        op::S_ADD64 | op::S_SUB64 => [S(Def, 2), S(Use, 2), S(Use, 2), None],
        op::S_SHL64 => [S(Def, 2), S(Use, 2), SAny, None],
        op::S_MEMTIME => [S(Def, 2), None, None, None],
        op::S_LAUNCH => [S(Def, 1), None, None, None],
        op::S_ADD..=op::S_ORN2 | op::S_SHL..=op::S_CMP_LE_U => [S(Def, 1), SAny, SAny, None],
        op::V_MOV | op::V_NOT | op::V_POPCNT..=op::V_BFREV | op::V_FLOOR..=op::V_FREXP_EXP
        | op::V_RCP..=op::V_CVT_I_F | op::V_CVT_F16_LO | op::V_CVT_F16_HI => [V(Def, 1), Any, None, None],
        op::V_BFI | op::V_PERM | op::V_FMA_F | op::V_CUBEID..=op::V_CUBEMA => [V(Def, 1), Any, Any, Any],
        op::V_ADD_CO | op::V_SUB_CO => [V(Def, 1), Any, Any, S(Def, 1)],
        op::V_ADDC | op::V_SUBB | op::V_CNDMASK => [V(Def, 1), Any, Any, S(Use, 1)],
        op::V_CMP_I..=op::V_CMP_F => [S(Def, 1), Any, Any, Raw],
        op::V_CMP_CLASS => [S(Def, 1), Any, Any, None],
        op::V_READLANE => [S(Def, 1), V(Use, 1), SAny, None],
        op::V_READFIRSTLANE => [S(Def, 1), V(Use, 1), None, None],
        op::V_WRITELANE => [V(DefUse, 1), SAny, SAny, None],
        op::V_PERMLANE => [V(Def, 1), V(Use, 1), Any, None],
        op::V_QUADPERM => [V(Def, 1), V(Use, 1), Any, None],
        op::V_MBCNT => [V(Def, 1), SAny, None, None],
        op::V_INTERP => [V(Def, 1), Any, Any, Raw],
        op::V_INTERP_FLAT => [V(Def, 1), None, None, Raw],
        op::V_ADD..=op::V_XOR | op::V_SHL..=op::V_BFE_I | op::V_MIN_I..=op::V_MAX_U | op::V_MORTON
        | op::V_ADD_F..=op::V_MUL_F | op::V_MIN_F | op::V_MAX_F | op::V_LDEXP | op::V_CVT_PK_F16 => {
            [V(Def, 1), Any, Any, None]
        }
        op::S_LOAD => [S(Def, scalar_load_dwords(hi >> 20)), OptS(1), S(Use, 2), None],
        op::S_BUFFER_LOAD => [S(Def, scalar_load_dwords(hi >> 20)), OptS(1), S(Use, 4), None],
        op::GLOBAL_LOAD | op::GLOBAL_STORE | op::GLOBAL_ATOMIC => {
            let data = match op {
                op::GLOBAL_LOAD => V(Def, size),
                op::GLOBAL_STORE => V(Use, size),
                _ => atomic_data(hi)?,
            };
            // A vector pair is the address without a scalar base; with one, a 32-bit offset.
            [data, OptV(2), OptS(2), None]
        }
        op::BUFFER_LOAD => [V(Def, size), OptV(1), S(Use, 4), None],
        op::BUFFER_STORE => [V(Use, size), OptV(1), S(Use, 4), None],
        op::BUFFER_ATOMIC => [atomic_data(hi)?, OptV(1), S(Use, 4), None],
        op::SCRATCH_LOAD | op::SHARED_LOAD => [V(Def, size), OptV(1), None, None],
        op::SCRATCH_STORE | op::SHARED_STORE => [V(Use, size), OptV(1), None, None],
        op::SHARED_ATOMIC => [atomic_data(hi)?, OptV(1), None, None],
        op::IMAGE_SAMPLE | op::IMAGE_FETCH => {
            let count = ((hi >> 8) & 15).count_ones() as u8;
            [V(Def, count), V(Use, texture_coordinates(hi)), S(Use, 8), S(Use, 8)]
        }
        op::EXP => [Raw, V(Use, 4), Raw, None],
        _ => return Err(format!("unknown opcode 0x{op:02x}")),
    })
}
fn atomic_data(hi: u32) -> Result<Kind, String> {
    let size = ((hi >> 20) & 3) as u8 + 1;
    let operation = (hi >> 24) & 15;
    if size > 2 || operation > 9 {
        return Err("reserved atomic size/operation".into());
    }
    let n = if operation == 2 { size * 2 } else { size };
    Ok(Kind::V(if hi >> 28 & 1 != 0 { Access::DefUse } else { Access::Use }, n))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inst {
    pub op: u8,
    pub d: u8,
    pub a: u8,
    pub b: u8,
    pub hi: u32,
}
impl Inst {
    pub fn new(op: u8) -> Self {
        Self { op, d: 0, a: 0, b: 0, hi: 0 }
    }
    pub fn field(self, f: usize) -> u8 {
        [self.d, self.a, self.b, self.hi as u8][f]
    }
    /// The literal occupies `hi` when a or b is 255 in the ALU formats.
    pub fn literal(self) -> bool {
        matches!(format(self.op), Some(Format::Salu | Format::Valu))
            && (self.a == LITERAL || self.b == LITERAL)
    }
    pub fn c_field(self) -> bool {
        matches!(format(self.op), Some(Format::Salu | Format::Valu | Format::Texture)) && !self.literal()
    }
    pub fn kinds(self) -> Result<[Kind; 4], String> {
        let mut k = fields(self.op, self.hi)?;
        if !self.c_field() {
            k[3] = Kind::None;
        }
        Ok(k)
    }
    /// Legality exactly as `Tooling/apex_isa` decodes (Docs/isa.md, Encoding).
    pub fn validate(self) -> Result<(), String> {
        let Some(fmt) = format(self.op) else {
            return Err(format!("unknown opcode 0x{:02x}", self.op));
        };
        let err = |m: &str| Err(m.to_string());
        let (d, a, b, hi) = (self.d, self.a, self.b, self.hi);
        let sreg = |x: u8| (SCALAR..EXEC).contains(&x);
        let vreg = |x: u8| x < VECTOR_REGISTERS;
        let sbase = |x: u8, n: u8, align: u8| sreg(x) && (x - SCALAR) % align == 0 && x - SCALAR + n <= SCALAR_REGISTERS;
        let vbase = |x: u8, n: u8, align: u8| vreg(x) && x % align == 0 && x as u32 + n as u32 <= VECTOR_REGISTERS as u32;
        let constant = |x: u8| (240..=254).contains(&x);
        match fmt {
            Format::Control => {
                let cond = matches!(self.op, op::S_CBRANCH_Z | op::S_CBRANCH_NZ);
                if d != 0 || b != 0 || (!cond && a != 0) {
                    return err("unused field nonzero");
                }
                if cond && !(sreg(a) || a == EXEC || constant(a)) {
                    return err("branch condition is not a scalar source");
                }
                let reserved = match self.op {
                    op::S_NOP | op::S_ENDPGM | op::S_BARRIER => hi != 0,
                    op::S_FENCE => hi >> 4 != 0,
                    op::S_SLEEP => hi >> 16 != 0,
                    _ => false,
                };
                if reserved {
                    return err("reserved control bits");
                }
            }
            Format::Salu | Format::Valu => return self.validate_alu(),
            Format::Memory => {
                let (size, policy, atom, ret) = ((hi >> 20) & 3, (hi >> 22) & 3, (hi >> 24) & 15, (hi >> 28) & 1);
                let atomic = matches!(self.op, op::GLOBAL_ATOMIC | op::BUFFER_ATOMIC | op::SHARED_ATOMIC);
                let smem = matches!(self.op, op::S_LOAD | op::S_BUFFER_LOAD);
                if hi >> 29 != 0 || policy == 3 || (!atomic && (atom | ret) != 0) || (atomic && atom > 9) {
                    return err("reserved memory bits");
                }
                let n = if smem { scalar_load_dwords(size) } else { size as u8 + 1 };
                if atomic {
                    if size > 1 || (self.op == op::SHARED_ATOMIC && size != 0) {
                        return err("atomic size out of range");
                    }
                    if !vbase(d, n * if atom == 2 { 2 } else { 1 }, n) {
                        return err("atomic data registers out of range or odd pair");
                    }
                } else if smem {
                    if !sbase(d, n, 1) {
                        return err("scalar destination out of range");
                    }
                } else if !vbase(d, n, 1) {
                    return err("vector data registers out of range");
                }
                let ok = match self.op {
                    op::S_LOAD => (a == LITERAL || sreg(a)) && sbase(b, 2, 2),
                    op::S_BUFFER_LOAD => (a == LITERAL || sreg(a)) && sbase(b, 4, 4),
                    op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC if b == LITERAL => vbase(a, 2, 2),
                    op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC => (a == LITERAL || vreg(a)) && sbase(b, 2, 2),
                    op::BUFFER_LOAD..=op::BUFFER_ATOMIC => (a == LITERAL || vreg(a)) && sbase(b, 4, 4),
                    _ => (a == LITERAL || vreg(a)) && b == 0,
                };
                if !ok {
                    return err("memory address operands");
                }
            }
            Format::Texture => {
                let (mask, gather, variant, dim) = ((hi >> 8) & 15, (hi >> 12) & 3, (hi >> 14) & 7, (hi >> 19) & 7);
                if hi >> 22 != 0 || mask == 0 || variant > TEX_GATHER || dim > DIM_CUBE_ARRAY
                    || (self.op == op::IMAGE_FETCH && variant != 0)
                    || (gather != 0 && variant != TEX_GATHER)
                {
                    return err("reserved texture bits");
                }
                if !vbase(d, mask.count_ones() as u8, 1) || !vbase(a, texture_coordinates(hi), 1)
                    || !sbase(b, 8, 4) || !sbase(hi as u8, 8, 4)
                {
                    return err("texture operands");
                }
            }
            Format::Export => {
                if hi != 0 || d > 9 || !vbase(a, 4, 1) || b > 3 {
                    return err("reserved export fields");
                }
            }
        }
        Ok(())
    }
    fn validate_alu(self) -> Result<(), String> {
        use Role::*;
        let (d, a, b, hi) = (self.d, self.a, self.b, self.hi);
        let roles = alu_roles(self.op);
        for (f, (role, x)) in roles.iter().zip([d, a, b]).enumerate() {
            match role {
                None if x != 0 => return Err(format!("unused field {} nonzero", ["d", "a", "b"][f])),
                None => {}
                r if !role_ok(*r, x, f < 3 && f > 0) => {
                    return Err(format!("reserved operand code {x} in field {}", ["d", "a", "b"][f]))
                }
                _ => {}
            }
        }
        if self.op == op::S_LAUNCH {
            return if hi > 6 { Err("launch selector out of range".into()) } else { Ok(()) };
        }
        if self.op == op::V_QUADPERM {
            return if a == LITERAL || hi >> 8 != 0 { Err("quad pattern".into()) } else { Ok(()) };
        }
        let literal = (roles[1] != None && a == LITERAL) || (roles[2] != None && b == LITERAL);
        if literal {
            return if roles[3] != None { Err("literal on an operation that uses c".into()) } else { Ok(()) };
        }
        let c = hi as u8;
        match roles[3] {
            None if c != 0 => return Err("unused field c nonzero".into()),
            None => {}
            r if !role_ok(r, c, false) => return Err(format!("reserved operand code {c} in field c")),
            _ => {}
        }
        let (modifiers, clamp) = ((hi >> 8) & 7 | (hi >> 11) & 7, (hi >> 14) & 1);
        if hi >> 15 != 0 || modifiers & !fp_sources(self.op) != 0 || (clamp != 0 && !fp_result(self.op)) {
            return Err("reserved ALU modifier bits".into());
        }
        Ok(())
    }
    pub fn encode(self) -> Result<u64, String> {
        self.validate()?;
        Ok(self.op as u64 | (self.d as u64) << 8 | (self.a as u64) << 16 | (self.b as u64) << 24
            | (self.hi as u64) << 32)
    }
    pub fn decode(x: u64) -> Result<Self, String> {
        let i = Self { op: x as u8, d: (x >> 8) as u8, a: (x >> 16) as u8, b: (x >> 24) as u8, hi: (x >> 32) as u32 };
        i.validate()?;
        Ok(i)
    }
    /// Registers read (false) or written (true), including every register of a group.
    pub fn registers(self, writes: bool) -> Vec<(Class, u8)> {
        let mut out = Vec::new();
        let kinds = self.kinds().unwrap();
        for (f, k) in kinds.iter().enumerate() {
            let code = self.field(f);
            let (class, access, n) = match *k {
                Kind::S(a, n) => (Class::S, a, n),
                Kind::V(a, n) => (Class::V, a, n),
                Kind::OptS(n) if code != LITERAL => (Class::S, Access::Use, n),
                Kind::OptV(n) if code != LITERAL => {
                    // A global offset is one register when a scalar base is present.
                    let n = if matches!(self.op, op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC) && self.b != LITERAL { 1 } else { n };
                    (Class::V, Access::Use, n)
                }
                Kind::Any | Kind::SAny if code < 224 => {
                    (if code < SCALAR { Class::V } else { Class::S }, Access::Use, 1)
                }
                _ => continue,
            };
            let hit = match access {
                Access::Def => writes,
                Access::Use => !writes,
                Access::DefUse => true,
            };
            if hit {
                let base = if class == Class::S { code - SCALAR } else { code };
                out.extend((0..n).map(|j| (class, base + j)));
            }
        }
        out
    }
    pub fn writes_exec(self) -> bool {
        self.d == EXEC && matches!(format(self.op), Some(Format::Salu))
            || matches!(self.op, op::S_AND_SAVEEXEC..=op::S_SETEXEC)
    }
    pub fn branch(self) -> bool {
        matches!(self.op, op::S_BRANCH..=op::S_CBRANCH_EXECNZ)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    None,
    /// Vector destination.
    Vd,
    /// Scalar destination, or an even scalar pair.
    Sd,
    Sd2,
    /// Scalar register.
    Sr,
    /// Vector source; scalar source; 64-bit scalar source.
    Vs,
    Ss,
    Ss2,
    /// Raw compare condition (6 or 8 values), attribute code, quad pattern.
    Cond6,
    Cond8,
    Attr,
    Pattern,
    /// `s_launch` selector in `hi`.
    Select,
}
/// ALU operand roles of d, a, b and c, as the reference decodes them.
fn alu_roles(o: u8) -> [Role; 4] {
    use Role::*;
    match o {
        op::S_MOV | op::S_NOT | op::S_FF1 | op::S_POPCNT | op::S_AND_SAVEEXEC..=op::S_ANDN2_SAVEEXEC => [Sd, Ss, None, None],
        op::S_CSELECT => [Sd, Ss, Ss, Ss],
        op::S_ADD64 | op::S_SUB64 => [Sd2, Ss2, Ss2, None],
        op::S_SHL64 => [Sd2, Ss2, Ss, None],
        op::S_SETEXEC => [None, Ss, None, None],
        op::S_MEMTIME => [Sd2, None, None, None],
        op::S_LAUNCH => [Sd, None, None, Select],
        0x20..=0x45 => [Sd, Ss, Ss, None],
        op::V_MOV | op::V_NOT | op::V_POPCNT..=op::V_BFREV | op::V_FLOOR..=op::V_FREXP_EXP
        | op::V_RCP..=op::V_CVT_I_F | op::V_CVT_F16_LO | op::V_CVT_F16_HI => [Vd, Vs, None, None],
        op::V_BFI | op::V_PERM | op::V_FMA_F | op::V_CUBEID..=op::V_CUBEMA => [Vd, Vs, Vs, Vs],
        op::V_ADD_CO | op::V_SUB_CO => [Vd, Vs, Vs, Sr],
        op::V_ADDC | op::V_SUBB | op::V_CNDMASK => [Vd, Vs, Vs, Ss],
        op::V_CMP_I | op::V_CMP_U => [Sd, Vs, Vs, Cond6],
        op::V_CMP_F => [Sd, Vs, Vs, Cond8],
        op::V_CMP_CLASS => [Sd, Vs, Vs, None],
        op::V_READLANE => [Sd, Vs, Ss, None],
        op::V_READFIRSTLANE => [Sd, Vs, None, None],
        op::V_WRITELANE => [Vd, Ss, Ss, None],
        op::V_QUADPERM => [Vd, Vs, Pattern, None],
        op::V_MBCNT => [Vd, Vs, None, None],
        op::V_INTERP => [Vd, Vs, Vs, Attr],
        op::V_INTERP_FLAT => [Vd, None, None, Attr],
        _ => [Vd, Vs, Vs, None],
    }
}
/// Whether `x` fills `role`; `literal` allows the literal (ALU sources a and b).
fn role_ok(role: Role, x: u8, literal: bool) -> bool {
    let sreg = (SCALAR..EXEC).contains(&x);
    let constant = (240..=254).contains(&x) || (x == LITERAL && literal);
    match role {
        Role::None | Role::Select => true,
        Role::Vd => x < VECTOR_REGISTERS,
        Role::Sd | Role::Sr => sreg,
        Role::Sd2 => sreg && (x - SCALAR) % 2 == 0,
        Role::Vs => x <= LANE || constant,
        Role::Ss => sreg || x == EXEC || constant,
        Role::Ss2 => (sreg && (x - SCALAR) % 2 == 0) || x == EXEC || constant,
        Role::Cond6 => x < 6,
        Role::Cond8 => x < 8,
        Role::Attr => x < 136,
        Role::Pattern => x == LITERAL,
    }
}

// Text form: `name d, a, b, c key:value...`; registers `v3`, `s4`, groups `v[4:7]`.
fn name(op: u8) -> String {
    NAMES.iter().find(|n| n.0 == op).map(|n| n.1.to_ascii_lowercase()).unwrap_or_default()
}
fn operand_text(code: u8, n: u8, literal: u32) -> String {
    match code {
        0..=127 if n > 1 => format!("v[{}:{}]", code, code + n - 1),
        0..=127 => format!("v{code}"),
        128..=223 if n > 1 => format!("s[{}:{}]", code - SCALAR, code - SCALAR + n - 1),
        128..=223 => format!("s{}", code - SCALAR),
        EXEC => "exec".into(),
        LANE => "lane".into(),
        240..=247 => format!("{}", INLINE[(code - 240) as usize] as i32),
        248..=254 => format!("{:?}", f32::from_bits(INLINE[(code - 240) as usize])),
        LITERAL => format!("0x{literal:08x}"),
        _ => format!("?{code}"),
    }
}
pub fn disassemble_one(i: Inst) -> String {
    let kinds = i.kinds().unwrap_or([Kind::None; 4]);
    let fmt = format(i.op).unwrap();
    let mut ops = Vec::new();
    for f in 0..4 {
        let code = i.field(f);
        let n = match kinds[f] {
            Kind::S(_, n) | Kind::V(_, n) => n,
            Kind::OptS(n) | Kind::OptV(n) => {
                if matches!(i.op, op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC) && f == 1 && i.b != LITERAL { 1 } else { n }
            }
            _ => 1,
        };
        let text = match kinds[f] {
            Kind::None => continue,
            Kind::Raw => format!("#{code}"),
            Kind::OptS(_) | Kind::OptV(_) if code == LITERAL => "off".into(),
            _ => {
                let mut t = operand_text(code, n, i.hi);
                if fmt == Format::Valu && !i.literal() && f > 0 && f < 4 {
                    if (i.hi >> (10 + f)) & 1 != 0 {
                        t = format!("|{t}|");
                    }
                    if (i.hi >> (7 + f)) & 1 != 0 {
                        t = format!("-{t}");
                    }
                }
                t
            }
        };
        ops.push(text);
    }
    let mut text = name(i.op);
    if !ops.is_empty() {
        text += " ";
        text += &ops.join(", ");
    }
    match fmt {
        Format::Valu if !i.literal() && (i.hi >> 14) & 1 != 0 => text += " clamp",
        Format::Salu if i.op == op::S_LAUNCH && i.hi != 0 => text += &format!(" imm:{}", i.hi),
        Format::Control if i.hi != 0 => {
            text += &format!(" {}:{}", if i.branch() { "offset" } else { "imm" }, i.hi as i32)
        }
        Format::Memory => {
            let offset = ((i.hi << 12) as i32) >> 12;
            if offset != 0 {
                text += &format!(" offset:{offset}");
            }
            text += &format!(" size:{}", (i.hi >> 20) & 3);
            for (k, shift, bits) in [("cache", 22, 3), ("atomic", 24, 15), ("return", 28, 1)] {
                if (i.hi >> shift) & bits != 0 {
                    text += &format!(" {k}:{}", (i.hi >> shift) & bits);
                }
            }
        }
        Format::Texture => {
            for (k, shift, bits) in [("mask", 8, 15), ("gather", 12, 3), ("variant", 14, 7),
                ("compare", 17, 1), ("offsets", 18, 1), ("dim", 19, 7)] {
                if (i.hi >> shift) & bits != 0 {
                    text += &format!(" {k}:{}", (i.hi >> shift) & bits);
                }
            }
        }
        _ => {}
    }
    text
}
pub fn disassemble(code: &[Inst]) -> String {
    code.iter().map(|&i| disassemble_one(i) + "\n").collect()
}

fn number(s: &str) -> Result<u32, String> {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |t| (true, t));
    let v = if let Some(x) = s.strip_prefix("0x") {
        u32::from_str_radix(x, 16).map_err(|_| format!("bad number {s}"))?
    } else if s.contains('.') || s.contains("inf") || s.contains("NaN") {
        let f: f32 = s.parse().map_err(|_| format!("bad float {s}"))?;
        return Ok(if neg { (-f).to_bits() } else { f.to_bits() });
    } else {
        s.parse::<u32>().map_err(|_| format!("bad number {s}"))?
    };
    Ok(if neg { v.wrapping_neg() } else { v })
}
/// Parses one operand; returns its code and any literal it needs.
fn parse_operand(text: &str) -> Result<(u8, Option<u32>, u32), String> {
    let (mut t, mut modifiers) = (text.trim(), 0);
    if let Some(r) = t.strip_prefix('-') {
        if r.starts_with(['v', 's', '|', 'e', 'l']) {
            t = r;
            modifiers |= 1;
        }
    }
    if t.starts_with('|') && t.ends_with('|') {
        t = &t[1..t.len() - 1];
        modifiers |= 2;
    }
    let register = |prefix: char, base: u8| -> Option<Result<u8, String>> {
        let r = t.strip_prefix(prefix)?;
        let first = r.strip_prefix('[').map_or(r, |g| g.split(':').next().unwrap_or(""));
        Some(first.parse::<u8>().map(|n| n + base).map_err(|_| format!("bad register {t}")))
    };
    let code = if t == "exec" {
        EXEC
    } else if t == "lane" {
        LANE
    } else if t == "off" {
        return Ok((LITERAL, None, 0));
    } else if let Some(r) = register('v', 0) {
        r?
    } else if let Some(r) = register('s', SCALAR) {
        r?
    } else if let Some(r) = t.strip_prefix('#') {
        return Ok((number(r)? as u8, None, 0));
    } else {
        let v = number(t)?;
        return Ok(match inline_code(v) {
            Some(c) => (c, None, modifiers),
            None => (LITERAL, Some(v), modifiers),
        });
    };
    Ok((code, None, modifiers))
}
pub fn assemble_one(line: &str) -> Result<Inst, String> {
    let line = line.trim();
    let (mnemonic, rest) = line.split_once(' ').unwrap_or((line, ""));
    let upper = mnemonic.to_ascii_uppercase();
    let opcode = NAMES.iter().find(|n| n.1 == upper).ok_or(format!("unknown mnemonic {mnemonic}"))?.0;
    let mut i = Inst::new(opcode);
    let fmt = format(opcode).unwrap();
    let mut operands = Vec::new();
    let mut keys = Vec::new();
    for token in rest.split(',').flat_map(|s| s.split_whitespace()) {
        let group = token.starts_with("v[") || token.starts_with("s[") || token.starts_with("-v[")
            || token.starts_with("|v[");
        if let Some((k, v)) = token.split_once(':').filter(|_| !group) {
            keys.push((k.to_string(), v.to_string()));
        } else if token == "clamp" {
            i.hi |= 1 << 14;
        } else {
            operands.push(token.to_string());
        }
    }
    // Field kinds depend on hi; apply keys first.
    for (k, v) in &keys {
        let v = if k == "offset" && fmt == Format::Memory {
            (number(v)? & 0xfffff) as u32
        } else {
            number(v)?
        };
        let (shift, mask) = match (fmt, k.as_str()) {
            (Format::Control, _) => (0, u32::MAX),
            (Format::Salu, "imm") if opcode == op::S_LAUNCH => (0, 7),
            (Format::Memory, "offset") => (0, 0xfffff),
            (Format::Memory, "size") => (20, 3),
            (Format::Memory, "cache") => (22, 3),
            (Format::Memory, "atomic") => (24, 15),
            (Format::Memory, "return") => (28, 1),
            (Format::Texture, "mask") => (8, 15),
            (Format::Texture, "gather") => (12, 3),
            (Format::Texture, "variant") => (14, 7),
            (Format::Texture, "compare") => (17, 1),
            (Format::Texture, "offsets") => (18, 1),
            (Format::Texture, "dim") => (19, 7),
            _ => return Err(format!("unknown key {k}")),
        };
        i.hi |= (v & mask) << shift;
    }
    let kinds = fields(opcode, i.hi)?;
    let mut next = operands.iter();
    let mut literal = None;
    for f in 0..4 {
        if kinds[f] == Kind::None {
            continue;
        }
        let Some(text) = next.next() else {
            if f == 3 && literal.is_some() {
                break;
            }
            return Err("missing operand".into());
        };
        let (code, lit, modifiers) = parse_operand(text)?;
        if let Some(l) = lit {
            if literal.is_some_and(|x| x != l) {
                return Err("two literals".into());
            }
            literal = Some(l);
        }
        if modifiers != 0 {
            i.hi |= (modifiers & 1) << (7 + f) | (modifiers >> 1) << (10 + f);
        }
        match f {
            0 => i.d = code,
            1 => i.a = code,
            2 => i.b = code,
            _ => i.hi = (i.hi & !0xff) | code as u32,
        }
    }
    if next.next().is_some() {
        return Err("extra operand".into());
    }
    if let Some(l) = literal {
        if i.hi & !0xff != 0 && matches!(fmt, Format::Salu | Format::Valu) {
            return Err("a literal excludes modifiers and clamp".into());
        }
        i.hi = l;
    }
    i.validate()?;
    Ok(i)
}
pub fn assemble(text: &str) -> Result<Vec<Inst>, String> {
    text.lines()
        .map(|l| l.split(';').next().unwrap().trim())
        .filter(|l| !l.is_empty())
        .map(assemble_one)
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Compute = 0,
    Vertex = 1,
    Fragment = 2,
}
pub const MAGIC: u32 = 0x5058_5041;
pub const HEADER: usize = 64;
// Fragment flags, header byte 5.
pub const EARLY_TESTS: u8 = 1;
pub const DISCARDS: u8 = 2;
pub const EXPORTS_DEPTH: u8 = 4;
pub const EXPORTS_SAMPLE_MASK: u8 = 8;
pub const SIDE_EFFECTS: u8 = 16;
pub const CENTROID: u8 = 32;
pub const INVERSE_W: u8 = 64;
pub const PER_SAMPLE: u8 = 128;

/// Program header (64 bytes, Docs/isa.md "Program header") and code.
#[derive(Clone, Debug)]
pub struct Program {
    pub stage: Stage,
    pub flags: u8,
    pub code: Vec<Inst>,
    pub entry: u32,
    pub local: [u32; 3],
    pub shared: u32,
    pub private: u32,
    /// Vertex output stride in bytes, or the fragment color-target mask with
    /// the dual-source flag in bit 8.
    pub output: u32,
    /// Fragment inputs: bit 7 flat; input k is vertex output varying k.
    pub inputs: Vec<u8>,
}
impl Program {
    pub fn new(stage: Stage, code: Vec<Inst>) -> Self {
        Self { stage, flags: 0, code, entry: 0, local: [16, 1, 1], shared: 0, private: 0, output: 0, inputs: Vec::new() }
    }
    pub fn invocations(&self) -> u32 {
        self.local.iter().product()
    }
    pub fn validate(&self) -> Result<(), String> {
        let n = self.local;
        let compute = self.stage == Stage::Compute;
        if self.code.is_empty()
            || self.code.len() > (1 << 20) / 8
            || self.entry as usize >= self.code.len()
            || !(1..=256).contains(&n[0]) || !(1..=256).contains(&n[1]) || !(1..=64).contains(&n[2])
            || !(1..=256).contains(&self.invocations())
            || self.shared > 32768
            || self.private % 4 != 0
            || (!compute && (n != [16, 1, 1] || self.shared != 0 || self.private != 0))
            || (self.stage != Stage::Fragment && (self.flags != 0 || !self.inputs.is_empty()))
            || (self.stage == Stage::Fragment && self.output >> 9 != 0)
            || (self.stage == Stage::Vertex && (self.output % 64 != 0 || self.output > 0xffff))
            || self.inputs.len() > 32 || self.inputs.iter().any(|&e| e & 0x7f != 0)
        {
            return Err("invalid program header".into());
        }
        for (pc, i) in self.code.iter().enumerate() {
            i.validate().map_err(|e| format!("instruction {pc}: {e}"))?;
            if i.branch() {
                let target = pc as i64 + 1 + i.hi as i32 as i64;
                if target < 0 || target >= self.code.len() as i64 {
                    return Err(format!("instruction {pc}: branch outside the program"));
                }
            }
        }
        Ok(())
    }
    pub fn bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut h = Vec::with_capacity(HEADER + 8 * self.code.len());
        h.extend(MAGIC.to_le_bytes());
        h.extend([self.stage as u8, self.flags, 0, 0]);
        h.extend((self.code.len() as u32 * 8).to_le_bytes());
        h.extend(self.entry.to_le_bytes());
        for x in [self.local[0], self.local[1], self.local[2], self.shared] {
            h.extend((x as u16).to_le_bytes());
        }
        h.extend(self.private.to_le_bytes());
        h.extend((self.output as u16).to_le_bytes());
        h.extend((self.inputs.len() as u16).to_le_bytes());
        h.extend(&self.inputs);
        h.resize(HEADER, 0);
        for i in &self.code {
            h.extend(i.encode()?.to_le_bytes());
        }
        Ok(h)
    }
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER {
            return Err("short header".into());
        }
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as u32;
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let stage = match b[4] {
            0 => Stage::Compute,
            1 => Stage::Vertex,
            2 => Stage::Fragment,
            _ => return Err("invalid stage".into()),
        };
        let count = u16_at(30) as usize;
        if u32_at(0) != MAGIC || u16_at(6) != 0 || u32_at(8) % 8 != 0 || count > 32
            || b[32 + count..HEADER].iter().any(|&x| x != 0)
            || u32_at(8) as u64 + HEADER as u64 != b.len() as u64
        {
            return Err("invalid program header".into());
        }
        let code = b[HEADER..]
            .chunks_exact(8)
            .map(|v| Inst::decode(u64::from_le_bytes(v.try_into().unwrap())))
            .collect::<Result<_, _>>()?;
        let p = Self {
            stage,
            flags: b[5],
            code,
            entry: u32_at(12),
            local: [u16_at(16), u16_at(18), u16_at(20)],
            shared: u16_at(22),
            private: u32_at(24),
            output: u16_at(28),
            inputs: b[32..32 + count].to_vec(),
        };
        p.validate()?;
        Ok(p)
    }
    pub fn text(&self) -> String {
        let mut t = format!(
            "stage:{} flags:{} entry:{} local:{}x{}x{} shared:{} private:{} output:{}",
            self.stage as u32, self.flags, self.entry, self.local[0], self.local[1], self.local[2],
            self.shared, self.private, self.output
        );
        for (i, e) in self.inputs.iter().enumerate() {
            t += &format!(" in{i}:{e}");
        }
        t + "\n" + &disassemble(&self.code)
    }
    pub fn from_text(text: &str) -> Result<Self, String> {
        let (head, body) = text.split_once('\n').unwrap_or((text, ""));
        let mut p = Self::new(Stage::Compute, assemble(body)?);
        for kv in head.split_whitespace() {
            let (k, v) = kv.split_once(':').ok_or("bad header key")?;
            match k {
                "stage" => {
                    p.stage = match number(v)? {
                        0 => Stage::Compute,
                        1 => Stage::Vertex,
                        2 => Stage::Fragment,
                        _ => return Err("bad stage".into()),
                    }
                }
                "flags" => p.flags = number(v)? as u8,
                "entry" => p.entry = number(v)?,
                "local" => {
                    let d: Vec<u32> = v.split('x').map(number).collect::<Result<_, _>>()?;
                    p.local = d.try_into().map_err(|_| "bad local size")?;
                }
                "shared" => p.shared = number(v)?,
                "private" => p.private = number(v)?,
                "output" => p.output = number(v)?,
                k if k.starts_with("in") => {
                    let i = k[2..].parse::<usize>().map_err(|_| "bad input")?.min(31);
                    if p.inputs.len() <= i {
                        p.inputs.resize(i + 1, 0);
                    }
                    p.inputs[i] = number(v)? as u8
                }
                _ => return Err(format!("unknown header key {k}")),
            }
        }
        p.validate()?;
        Ok(p)
    }
}
