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

pub fn fp_source(op: u8) -> bool {
    matches!(op, op::V_CMP_F | op::V_CMP_CLASS | op::V_ADD_F..=op::V_CVT_PK_F16 | op::V_CUBEID..=op::V_CUBEMA)
        && !matches!(op, op::V_CVT_F_U | op::V_CVT_F_I)
}
pub fn fp_result(op: u8) -> bool {
    matches!(op, op::V_ADD_F..=op::V_FRACT | op::V_FREXP_MANT | op::V_LDEXP..=op::V_CVT_F_I
        | op::V_CVT_F16_LO | op::V_CVT_F16_HI | op::V_CUBESC..=op::V_CUBEMA | op::V_INTERP
        | op::V_INTERP_FLAT)
}

pub fn scalar_load_dwords(code: u32) -> u8 {
    [1, 2, 4, 8][code as usize & 3]
}
pub fn texture_coordinates(hi: u32, fetch: bool) -> u8 {
    let dim = (hi >> 19) & 7;
    let variant = (hi >> 14) & 7;
    let (coords, grads) = match dim {
        DIM_1D => (1, 1),
        DIM_2D => (2, 2),
        DIM_3D => (3, 3),
        DIM_1D_ARRAY => (2, 1),
        _ => (3, 2),
    };
    if fetch {
        return coords + 1;
    }
    coords
        + matches!(variant, TEX_LEVEL | TEX_BIAS) as u8
        + ((hi >> 17) & 1) as u8
        + ((hi >> 18) & 1) as u8
        + if variant == TEX_GRADIENT { 2 * grads } else { 0 }
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
        op::S_LAUNCH => [S(Def, 1), SAny, None, None],
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
        op::SCRATCH_LOAD => [V(Def, size), OptV(1), S(Use, 2), None],
        op::SCRATCH_STORE => [V(Use, size), OptV(1), S(Use, 2), None],
        op::SHARED_LOAD => [V(Def, size), OptV(1), OptS(1), None],
        op::SHARED_STORE => [V(Use, size), OptV(1), OptS(1), None],
        op::SHARED_ATOMIC => [atomic_data(hi)?, OptV(1), OptS(1), None],
        op::IMAGE_SAMPLE => {
            let count = ((hi >> 8) & 15).count_ones() as u8;
            let sampler = if (hi >> 14) & 7 == TEX_FETCH { None } else { S(Use, 8) };
            [V(Def, count), V(Use, texture_coordinates(hi, false)), S(Use, 8), sampler]
        }
        op::IMAGE_FETCH => {
            let count = ((hi >> 8) & 15).count_ones() as u8;
            [V(Def, count), V(Use, texture_coordinates(hi, true)), S(Use, 8), None]
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
    pub fn validate(self) -> Result<(), String> {
        let kinds = fields(self.op, self.hi)?;
        let fmt = format(self.op).unwrap();
        let literal = self.literal();
        if literal && kinds[3] != Kind::None {
            return Err("literal in an instruction that needs c".into());
        }
        for f in 0..3 {
            let code = self.field(f);
            check_operand(kinds[f], code, f)?;
            let alu_source = matches!(fmt, Format::Salu | Format::Valu) && (f == 1 || f == 2)
                && matches!(kinds[f], Kind::Any | Kind::SAny);
            if code == LITERAL && !alu_source && !matches!(kinds[f], Kind::OptS(_) | Kind::OptV(_) | Kind::Raw) {
                return Err("literal outside an ALU a/b source".into());
            }
        }
        if matches!(self.op, op::GLOBAL_LOAD..=op::GLOBAL_ATOMIC) && self.b == LITERAL && self.a != LITERAL
            && (self.a % 2 != 0 || self.a >= VECTOR_REGISTERS - 1)
        {
            return Err("global address pair".into());
        }
        match fmt {
            Format::Salu | Format::Valu if !literal => {
                check_operand(kinds[3], self.hi as u8, 3)?;
                if self.hi as u8 == LITERAL && kinds[3] != Kind::Raw {
                    return Err("literal in c".into());
                }
                let modifiers = (self.hi >> 8) & 0x3f;
                let clamp = (self.hi >> 14) & 1;
                if self.hi >> 15 != 0
                    || (modifiers != 0 && !fp_source(self.op))
                    || (clamp != 0 && !fp_result(self.op))
                    || (fmt == Format::Salu && (modifiers | clamp) != 0)
                {
                    return Err("reserved ALU modifier bits".into());
                }
            }
            Format::Valu | Format::Salu => {}
            Format::Control => {
                if (self.op == op::S_FENCE && self.hi >> 4 != 0)
                    || (matches!(self.op, op::S_NOP | op::S_ENDPGM | op::S_BARRIER) && self.hi != 0)
                {
                    return Err("reserved control bits".into());
                }
            }
            Format::Memory => {
                if self.hi >> 29 != 0
                    || (!matches!(self.op, op::GLOBAL_ATOMIC | op::BUFFER_ATOMIC | op::SHARED_ATOMIC)
                        && (self.hi >> 24) & 31 != 0)
                    || (self.hi >> 22) & 3 == 3
                {
                    return Err("reserved memory bits".into());
                }
            }
            Format::Texture => {
                let variant = (self.hi >> 14) & 7;
                let dim = (self.hi >> 19) & 7;
                if self.hi >> 22 != 0
                    || variant > TEX_GATHER
                    || dim > DIM_CUBE_ARRAY
                    || (self.hi >> 8) & 15 == 0
                    || (self.op == op::IMAGE_FETCH && (variant != TEX_FETCH || self.hi & 0x630ff != 0))
                    || (self.op == op::IMAGE_SAMPLE && variant == TEX_FETCH)
                    || (variant != TEX_GATHER && (self.hi >> 12) & 3 != 0)
                    || (variant == TEX_GATHER && ((self.hi >> 8) & 15).count_ones() != 4)
                {
                    return Err("reserved texture bits".into());
                }
            }
            Format::Export => {
                if self.d > 9 || self.b > 3 || self.hi != 0 {
                    return Err("reserved export fields".into());
                }
            }
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

fn check_operand(kind: Kind, code: u8, field: usize) -> Result<(), String> {
    let scalar = |n: u8| code >= SCALAR && code < EXEC && (code - SCALAR) + n <= SCALAR_REGISTERS
        && (n == 1 || (code - SCALAR) % 2 == 0);
    let vector = |n: u8| code < VECTOR_REGISTERS && code + n <= VECTOR_REGISTERS
        && (n != 2 || code % 2 == 0);
    let constant = code >= 240;
    let ok = match kind {
        Kind::None => code == 0,
        Kind::Raw => true,
        Kind::S(Access::Def, 1) if field == 0 => scalar(1) || code == EXEC,
        Kind::S(_, n) => scalar(n),
        Kind::V(_, n) => vector(n),
        Kind::Any => code < VECTOR_REGISTERS || scalar(1) || code == EXEC || code == LANE || constant,
        Kind::SAny => scalar(1) || code == EXEC || constant,
        Kind::OptS(n) => code == LITERAL || scalar(n),
        Kind::OptV(_) => code == LITERAL || code < VECTOR_REGISTERS,
    };
    if ok {
        Ok(())
    } else {
        Err(format!("reserved operand code {code} in field {}", ["d", "a", "b", "c"][field]))
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
// Fragment flags, header word 1 bits 15:8.
pub const EARLY_TESTS: u8 = 1;
pub const DISCARDS: u8 = 2;
pub const EXPORTS_DEPTH: u8 = 4;
pub const EXPORTS_SAMPLE_MASK: u8 = 8;
pub const SIDE_EFFECTS: u8 = 16;
pub const CENTROID: u8 = 32;
pub const INVERSE_W: u8 = 64;
pub const PER_SAMPLE: u8 = 128;

/// Program header (64 bytes) and code.
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
    /// Fragment inputs: bit 7 valid, bit 6 flat, bits 5:0 vertex output location.
    pub inputs: [u8; 32],
}
impl Program {
    pub fn new(stage: Stage, code: Vec<Inst>) -> Self {
        Self { stage, flags: 0, code, entry: 0, local: [16, 1, 1], shared: 0, private: 0, output: 0, inputs: [0; 32] }
    }
    pub fn invocations(&self) -> u32 {
        self.local.iter().product()
    }
    pub fn validate(&self) -> Result<(), String> {
        let n = self.local;
        if self.code.is_empty()
            || self.code.len() > (1 << 20) / 8
            || self.entry as usize >= self.code.len()
            || n[0] > 256 || n[1] > 256 || n[2] > 64
            || !(1..=256).contains(&self.invocations())
            || self.shared > 32768
            || self.private % 4 != 0
            || (self.stage != Stage::Compute && (self.invocations() != 16 || self.shared != 0))
            || (self.stage != Stage::Fragment && (self.flags != 0 || self.inputs != [0; 32]))
            || (self.stage == Stage::Fragment && self.output >> 9 != 0)
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
        let mut h = [0u32; 16];
        h[0] = MAGIC;
        h[1] = self.stage as u32 | (self.flags as u32) << 8;
        h[2] = self.code.len() as u32 * 8;
        h[3] = self.entry;
        h[4] = self.local[0] | self.local[1] << 9 | self.local[2] << 18;
        h[5] = self.shared;
        h[6] = self.private;
        h[7] = self.output;
        for (i, &e) in self.inputs.iter().enumerate() {
            h[8 + i / 4] |= (e as u32) << (8 * (i % 4));
        }
        let mut out: Vec<u8> = h.iter().flat_map(|x| x.to_le_bytes()).collect();
        for i in &self.code {
            out.extend(i.encode()?.to_le_bytes());
        }
        Ok(out)
    }
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER {
            return Err("short header".into());
        }
        let h: Vec<u32> = b[..HEADER].chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
        let stage = match h[1] & 0xff {
            0 => Stage::Compute,
            1 => Stage::Vertex,
            2 => Stage::Fragment,
            _ => return Err("invalid stage".into()),
        };
        if h[0] != MAGIC || h[1] >> 16 != 0 || h[2] % 8 != 0 || h[4] >> 25 != 0
            || h[2] as u64 + HEADER as u64 != b.len() as u64
        {
            return Err("invalid program header".into());
        }
        let code = b[HEADER..]
            .chunks_exact(8)
            .map(|v| Inst::decode(u64::from_le_bytes(v.try_into().unwrap())))
            .collect::<Result<_, _>>()?;
        let mut inputs = [0u8; 32];
        for (i, e) in inputs.iter_mut().enumerate() {
            *e = (h[8 + i / 4] >> (8 * (i % 4))) as u8;
        }
        let p = Self {
            stage,
            flags: (h[1] >> 8) as u8,
            code,
            entry: h[3],
            local: [h[4] & 0x1ff, (h[4] >> 9) & 0x1ff, h[4] >> 18],
            shared: h[5],
            private: h[6],
            output: h[7],
            inputs,
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
        for (i, e) in self.inputs.iter().enumerate().filter(|e| *e.1 != 0) {
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
                    p.inputs[k[2..].parse::<usize>().map_err(|_| "bad input")?.min(31)] = number(v)? as u8
                }
                _ => return Err(format!("unknown header key {k}")),
            }
        }
        p.validate()?;
        Ok(p)
    }
}
