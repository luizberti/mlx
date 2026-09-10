//! Hand-written core of the Zig MLX wrapper: handles, errors, scalar/data marshalling.
//! The op surface (mlx.add, mlx.linalg.*, ...) is generated at build time and re-exports this.
const std = @import("std");

/// Raw mlx-c FFI, for anything not yet wrapped.
pub const c = @import("c");

pub const Error = error{Mlx};

threadlocal var error_len: usize = 0;
threadlocal var error_buf: [4096]u8 = undefined; // JIT failures quote the compiler; keep them whole

fn onError(msg: [*c]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    const s = std.mem.span(@as([*:0]const u8, @ptrCast(msg)));
    error_len = @min(s.len, error_buf.len);
    @memcpy(error_buf[0..error_len], s[0..error_len]);
}

/// Install the error handler that captures messages for lastError().
/// Call once at startup: mlx-c's default handler prints the message and
/// exit(-1)s, so until this runs no failed call ever returns error.Mlx.
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

/// The override installed by setMetallibPath, copied into `buf`; empty when none is set
/// (MLX then looks next to the executable). Fails on builds without the Metal backend.
pub fn getMetallibPath(buf: []u8) (Error || error{NoSpaceLeft})![]const u8 {
    var s = c.mlx_string_new();
    defer _ = c.mlx_string_free(s);
    try check(c.mlx_metal_get_metallib_path(&s));
    const path = std.mem.span(c.mlx_string_data(s));
    if (path.len > buf.len) return error.NoSpaceLeft;
    @memcpy(buf[0..path.len], path);
    return buf[0..path.len];
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

pub const Device = enum(c_uint) {
    cpu = c.MLX_CPU,
    gpu = c.MLX_GPU,
};

/// Streams are thread-affine: MLX registers a stream's command encoder on the thread that
/// created it, so evaluating on it from another thread fails with "There is no Stream(...)
/// in current thread". Create per thread, or use threadUnsafe() for one shared across threads.
pub const Stream = extern struct {
    h: c.mlx_stream,

    /// This thread's default GPU stream. Fails without a GPU backend, or when the Metal
    /// library can't be loaded (see setMetallibPath).
    pub fn gpu() Error!Stream {
        return wrap(c.mlx_default_gpu_stream_new());
    }

    /// This thread's default CPU stream.
    pub fn cpu() Error!Stream {
        return wrap(c.mlx_default_cpu_stream_new());
    }

    /// A new stream on `device`, bound to the calling thread like the defaults.
    pub fn init(device: Device) Error!Stream {
        const dev = c.mlx_device_new_type(@intFromEnum(device), 0);
        defer _ = c.mlx_device_free(dev);
        return wrap(c.mlx_stream_new_device(dev));
    }

    /// A new stream on `device` usable from any thread: registered globally instead of per
    /// thread. MLX applies no synchronization to it; data races on it are the caller's.
    pub fn threadUnsafe(device: Device) Error!Stream {
        const dev = c.mlx_device_new_type(@intFromEnum(device), 0);
        defer _ = c.mlx_device_free(dev);
        return wrap(c.mlx_stream_new_thread_unsafe(dev));
    }

    /// Block until everything queued on this stream has run; surfaces deferred errors.
    pub fn synchronize(self: Stream) Error!void {
        try check(c.mlx_synchronize(self.h));
    }

    pub fn deinit(self: Stream) void {
        _ = c.mlx_stream_free(self.h);
    }

    // mlx-c reports constructor failures through the error handler and hands back a null
    // handle; surface that here instead of letting the next op fail with a confusing message.
    fn wrap(h: c.mlx_stream) Error!Stream {
        if (h.ctx == null) return error.Mlx;
        return .{ .h = h };
    }
};
