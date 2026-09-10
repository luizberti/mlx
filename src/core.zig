//! Hand-written core of the Zig MLX wrapper: handles, errors, scalar/data marshalling.
//! The op surface (mlx.add, mlx.linalg.*, ...) is generated at build time and re-exports this.
const std = @import("std");

/// Raw mlx-c FFI, for anything not yet wrapped.
pub const c = @import("c");

pub const Error = error{Mlx};

threadlocal var error_len: usize = 0;
threadlocal var error_buf: [512]u8 = undefined;

fn onError(msg: [*c]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    const s = std.mem.span(@as([*:0]const u8, @ptrCast(msg)));
    error_len = @min(s.len, error_buf.len);
    @memcpy(error_buf[0..error_len], s[0..error_len]);
}

/// Install the error handler that captures messages for lastError().
/// Call once at startup; without it failed calls still return error.Mlx,
/// but messages go to stderr (mlx-c default) instead of lastError().
pub fn init() void {
    c.mlx_set_error_handler(onError, null, null);
}

/// Message from the most recent failed mlx call on this thread.
pub fn lastError() []const u8 {
    return error_buf[0..error_len];
}

pub fn check(status: c_int) Error!void {
    if (status != 0) return error.Mlx;
}

pub fn metalAvailable() bool {
    var res = false;
    _ = c.mlx_metal_is_available(&res);
    return res;
}

/// Override where the Metal backend loads mlx.metallib from. Must be called before the
/// first GPU op; without it MLX looks next to the executable (see README).
pub fn setMetallibPath(path: [:0]const u8) Error!void {
    try check(c.mlx_metal_set_metallib_path(path));
}

pub const DType = enum(c_uint) {
    bool = 0,
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    bfloat16,
    complex64,

    pub fn of(comptime T: type) DType {
        return switch (T) {
            bool => .bool,
            u8 => .uint8,
            u16 => .uint16,
            u32 => .uint32,
            u64 => .uint64,
            i8 => .int8,
            i16 => .int16,
            i32 => .int32,
            i64 => .int64,
            f16 => .float16,
            f32 => .float32,
            f64 => .float64,
            else => @compileError("no MLX dtype for " ++ @typeName(T)),
        };
    }

    pub fn size(self: DType) usize {
        return c.mlx_dtype_size(@intFromEnum(self));
    }
};

/// File-as-struct; at build time gen.zig appends generated op methods to it.
pub const Array = @import("Array.zig");

/// Owned mlx_vector_array. Used for variadic inputs/outputs (split, vjp, ...).
pub const Arrays = extern struct {
    h: c.mlx_vector_array,

    pub fn init() Arrays {
        return .{ .h = c.mlx_vector_array_new() };
    }

    /// Copies the handles (the underlying arrays are refcounted, not copied).
    pub fn fromSlice(items: []const Array) Arrays {
        return .{ .h = c.mlx_vector_array_new_data(@ptrCast(items.ptr), items.len) };
    }

    pub fn deinit(self: Arrays) void {
        _ = c.mlx_vector_array_free(self.h);
    }

    pub fn len(self: Arrays) usize {
        return c.mlx_vector_array_size(self.h);
    }

    /// Returns a new owned handle; deinit it.
    pub fn at(self: Arrays, i: usize) Error!Array {
        var r = c.mlx_array_new();
        try check(c.mlx_vector_array_get(&r, self.h, i));
        return .{ .info = .{ .handle = r } };
    }

    pub fn append(self: Arrays, a: Array) Error!void {
        try check(c.mlx_vector_array_append_value(self.h, a.info.handle));
    }
};

pub const Stream = extern struct {
    h: c.mlx_stream,

    pub fn gpu() Stream {
        return .{ .h = c.mlx_default_gpu_stream_new() };
    }

    pub fn cpu() Stream {
        return .{ .h = c.mlx_default_cpu_stream_new() };
    }

    pub fn deinit(self: Stream) void {
        _ = c.mlx_stream_free(self.h);
    }
};
