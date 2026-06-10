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

fn itemFn(comptime T: type) fn ([*c]T, c.mlx_array) callconv(.c) c_int {
    return switch (T) {
        bool => c.mlx_array_item_bool,
        u8 => c.mlx_array_item_uint8,
        u16 => c.mlx_array_item_uint16,
        u32 => c.mlx_array_item_uint32,
        u64 => c.mlx_array_item_uint64,
        i8 => c.mlx_array_item_int8,
        i16 => c.mlx_array_item_int16,
        i32 => c.mlx_array_item_int32,
        i64 => c.mlx_array_item_int64,
        f16 => c.mlx_array_item_float16,
        f32 => c.mlx_array_item_float32,
        f64 => c.mlx_array_item_float64,
        else => @compileError("no MLX item accessor for " ++ @typeName(T)),
    };
}

fn dataFn(comptime T: type) fn (c.mlx_array) callconv(.c) [*c]const T {
    return switch (T) {
        bool => c.mlx_array_data_bool,
        u8 => c.mlx_array_data_uint8,
        u16 => c.mlx_array_data_uint16,
        u32 => c.mlx_array_data_uint32,
        u64 => c.mlx_array_data_uint64,
        i8 => c.mlx_array_data_int8,
        i16 => c.mlx_array_data_int16,
        i32 => c.mlx_array_data_int32,
        i64 => c.mlx_array_data_int64,
        f16 => c.mlx_array_data_float16,
        f32 => c.mlx_array_data_float32,
        f64 => c.mlx_array_data_float64,
        else => @compileError("no MLX data accessor for " ++ @typeName(T)),
    };
}

pub const Array = extern struct {
    h: c.mlx_array,

    comptime {
        std.debug.assert(@sizeOf(Array) == @sizeOf(c.mlx_array));
        std.debug.assert(@alignOf(Array) == @alignOf(c.mlx_array));
    }

    pub fn scalar(v: anytype) Array {
        return .{ .h = switch (@TypeOf(v)) {
            bool => c.mlx_array_new_bool(v),
            comptime_int, i8, i16, i32, u8, u16 => c.mlx_array_new_int(@intCast(v)),
            comptime_float, f32 => c.mlx_array_new_float32(v),
            f64 => c.mlx_array_new_float64(v),
            else => @compileError("no MLX scalar constructor for " ++ @typeName(@TypeOf(v))),
        } };
    }

    /// Copies `items`. shape products must equal items.len.
    pub fn fromSlice(comptime T: type, items: []const T, shape_: []const i32) Array {
        return .{ .h = c.mlx_array_new_data(
            @ptrCast(items.ptr),
            shape_.ptr,
            @intCast(shape_.len),
            @intFromEnum(DType.of(T)),
        ) };
    }

    pub fn deinit(self: Array) void {
        _ = c.mlx_array_free(self.h);
    }

    /// Rebind this handle to src's value (mlx_array_set).
    pub fn assign(self: *Array, src: Array) Error!void {
        try check(c.mlx_array_set(&self.h, src.h));
    }

    /// New owned handle to the same underlying array — a refcount bump
    /// (mlx_array_new + mlx_array_set; mlx-c has no explicit retain), NOT a
    /// data copy. For a graph-level copy of the data, use the mlx.copy op.
    pub fn clone(self: Array) Error!Array {
        var h = c.mlx_array_new();
        check(c.mlx_array_set(&h, self.h)) catch |err| {
            _ = c.mlx_array_free(h);
            return err;
        };
        return .{ .h = h };
    }

    pub fn ndim(self: Array) usize {
        return c.mlx_array_ndim(self.h);
    }

    pub fn size(self: Array) usize {
        return c.mlx_array_size(self.h);
    }

    pub fn shape(self: Array) []const i32 {
        const n = c.mlx_array_ndim(self.h);
        return if (n == 0) &.{} else c.mlx_array_shape(self.h)[0..n];
    }

    pub fn dtype(self: Array) DType {
        return @enumFromInt(c.mlx_array_dtype(self.h));
    }

    pub fn eval(self: Array) Error!void {
        try check(c.mlx_array_eval(self.h));
    }

    /// Evaluates and copies the single element out.
    pub fn item(self: Array, comptime T: type) Error!T {
        var v: T = undefined;
        try check(itemFn(T)(&v, self.h));
        return v;
    }

    /// View of the evaluated array's buffer; call eval() first.
    /// Valid while the array is alive and unmodified.
    pub fn data(self: Array, comptime T: type) Error![]const T {
        const p = dataFn(T)(self.h);
        if (p == null) return error.Mlx;
        return p[0..c.mlx_array_size(self.h)];
    }

    pub fn format(self: Array, w: *std.Io.Writer) std.Io.Writer.Error!void {
        var s = c.mlx_string_new();
        defer _ = c.mlx_string_free(s);
        if (c.mlx_array_tostring(&s, self.h) != 0) return w.writeAll("<invalid mlx.Array>");
        try w.writeAll(std.mem.span(c.mlx_string_data(s)));
    }
};

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
        return .{ .h = r };
    }

    pub fn append(self: Arrays, a: Array) Error!void {
        try check(c.mlx_vector_array_append_value(self.h, a.h));
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
