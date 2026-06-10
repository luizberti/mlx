//! Function transforms: closures over Zig functions, autodiff, eager eval, compile.
const core = @import("core.zig");
const c = core.c;
const Error = core.Error;
const check = core.check;
const Array = core.Array;
const Arrays = core.Arrays;

pub fn eval(outputs: Arrays) Error!void {
    try check(c.mlx_eval(outputs.h));
}

pub fn asyncEval(outputs: Arrays) Error!void {
    try check(c.mlx_async_eval(outputs.h));
}

pub const Closure = extern struct {
    h: c.mlx_closure,

    /// Wrap a plain Zig fn. The input Arrays is borrowed (don't deinit it);
    /// ownership of the returned Arrays transfers to MLX.
    pub fn init(comptime f: fn (Arrays) Error!Arrays) Closure {
        const W = struct {
            fn cb(res: [*c]c.mlx_vector_array, input: c.mlx_vector_array) callconv(.c) c_int {
                const out = f(.{ .h = input }) catch return 1;
                defer out.deinit();
                return c.mlx_vector_array_set(res, out.h);
            }
        };
        return .{ .h = c.mlx_closure_new_func(W.cb) };
    }

    /// Wrap a stateful Zig fn. `ctx` must outlive the closure; it is not freed by it.
    pub fn initCtx(comptime Ctx: type, comptime f: fn (*Ctx, Arrays) Error!Arrays, ctx: *Ctx) Closure {
        const W = struct {
            fn cb(res: [*c]c.mlx_vector_array, input: c.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                const p: *Ctx = @ptrCast(@alignCast(payload.?));
                const out = f(p, .{ .h = input }) catch return 1;
                defer out.deinit();
                return c.mlx_vector_array_set(res, out.h);
            }
        };
        return .{ .h = c.mlx_closure_new_func_payload(W.cb, ctx, null) };
    }

    /// Wrap a single-array Zig fn.
    pub fn unary(comptime f: fn (Array) Error!Array) Closure {
        const W = struct {
            fn cb(res: [*c]c.mlx_array, x: c.mlx_array) callconv(.c) c_int {
                const out = f(.{ .info = .{ .handle = x } }) catch return 1;
                defer out.deinit();
                return c.mlx_array_set(res, out.info.handle);
            }
        };
        return .{ .h = c.mlx_closure_new_unary(W.cb) };
    }

    pub fn deinit(self: Closure) void {
        _ = c.mlx_closure_free(self.h);
    }

    pub fn apply(self: Closure, input: Arrays) Error!Arrays {
        var res = c.mlx_vector_array_new();
        try check(c.mlx_closure_apply(&res, self.h, input.h));
        return .{ .h = res };
    }
};

/// Returns .{ outputs, vjps } — gradients wrt primals, weighted by cotangents.
pub fn vjp(fun: Closure, primals: Arrays, cotangents: Arrays) Error!struct { Arrays, Arrays } {
    var r0 = c.mlx_vector_array_new();
    var r1 = c.mlx_vector_array_new();
    try check(c.mlx_vjp(&r0, &r1, fun.h, primals.h, cotangents.h));
    return .{ .{ .h = r0 }, .{ .h = r1 } };
}

/// Returns .{ outputs, jvps }.
pub fn jvp(fun: Closure, primals: Arrays, tangents: Arrays) Error!struct { Arrays, Arrays } {
    var r0 = c.mlx_vector_array_new();
    var r1 = c.mlx_vector_array_new();
    try check(c.mlx_jvp(&r0, &r1, fun.h, primals.h, tangents.h));
    return .{ .{ .h = r0 }, .{ .h = r1 } };
}

pub const ValueAndGrad = extern struct {
    h: c.mlx_closure_value_and_grad,

    pub fn deinit(self: ValueAndGrad) void {
        _ = c.mlx_closure_value_and_grad_free(self.h);
    }

    /// Returns .{ values, gradients } — gradients wrt the argnums it was built with.
    pub fn apply(self: ValueAndGrad, input: Arrays) Error!struct { Arrays, Arrays } {
        var r0 = c.mlx_vector_array_new();
        var r1 = c.mlx_vector_array_new();
        try check(c.mlx_closure_value_and_grad_apply(&r0, &r1, self.h, input.h));
        return .{ .{ .h = r0 }, .{ .h = r1 } };
    }
};

pub fn valueAndGrad(fun: Closure, argnums: []const i32) Error!ValueAndGrad {
    var res = c.mlx_closure_value_and_grad_new();
    try check(c.mlx_value_and_grad(&res, fun.h, argnums.ptr, argnums.len));
    return .{ .h = res };
}

pub fn checkpoint(fun: Closure) Error!Closure {
    var res = c.mlx_closure_new();
    try check(c.mlx_checkpoint(&res, fun.h));
    return .{ .h = res };
}

pub fn compile(fun: Closure, shapeless: bool) Error!Closure {
    var res = c.mlx_closure_new();
    try check(c.mlx_compile(&res, fun.h, shapeless));
    return .{ .h = res };
}
