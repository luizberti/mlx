//! mal — MLX for Zig. Module root: at build time gen.zig appends the generated
//! op wrappers (ops at top level, linalg/fft/random as namespaces) to this file.
const core = @import("core.zig");
const transforms = @import("transforms.zig");

/// Raw mlx-c FFI, for anything not wrapped.
pub const cffi = core.c;
pub const Error = core.Error;
pub const DType = core.DType;
pub const Array = core.Array;
pub const Arrays = core.Arrays;
pub const Stream = core.Stream;
pub const Scope = @import("Scope.zig");
pub const init = core.init;
pub const lastError = core.lastError;
pub const check = core.check;
pub const metalAvailable = core.metalAvailable;
pub const setMetallibPath = core.setMetallibPath;

pub const Closure = transforms.Closure;
pub const ValueAndGrad = transforms.ValueAndGrad;
pub const eval = transforms.eval;
pub const asyncEval = transforms.asyncEval;
pub const vjp = transforms.vjp;
pub const jvp = transforms.jvp;
pub const valueAndGrad = transforms.valueAndGrad;
pub const checkpoint = transforms.checkpoint;
pub const compile = transforms.compile;
