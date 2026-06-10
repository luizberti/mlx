//! # MLX Array

const stdz = @import("std");
const core = @import("core.zig");
const root = @import("root.zig");
const cffi = core.c;
const check = core.check;
const Error = core.Error;
const Arrays = core.Arrays;
const DType = core.DType;
const Stream = core.Stream;

info: Info,

comptime {
    // Arrays.fromSlice reinterprets []const Array as []const mlx_array.
    stdz.debug.assert(@sizeOf(@This()) == @sizeOf(cffi.mlx_array));
    stdz.debug.assert(@alignOf(@This()) == @alignOf(cffi.mlx_array));
}

pub const Info = extern struct {
    handle: cffi.mlx_array,

    pub fn ndim(self: @This()) usize {
        return cffi.mlx_array_ndim(self.handle);
    }

    pub fn size(self: @This()) usize {
        return cffi.mlx_array_size(self.handle);
    }

    pub fn shape(self: @This()) []const i32 {
        const n = cffi.mlx_array_ndim(self.handle);
        return if (n == 0) &.{} else cffi.mlx_array_shape(self.handle)[0..n];
    }

    pub fn dtype(self: @This()) DType {
        return @enumFromInt(cffi.mlx_array_dtype(self.handle));
    }
};

pub fn scalar(v: anytype) @This() {
    return .{ .info = .{ .handle = switch (@TypeOf(v)) {
        bool => cffi.mlx_array_new_bool(v),
        comptime_int, i8, i16, i32, u8, u16 => cffi.mlx_array_new_int(@intCast(v)),
        comptime_float, f32 => cffi.mlx_array_new_float32(v),
        f64 => cffi.mlx_array_new_float64(v),
        else => @compileError("no MLX scalar constructor for " ++ @typeName(@TypeOf(v))),
    } } };
}

/// Copies `items`. shape products must equal items.len.
pub fn fromSlice(comptime T: type, items: []const T, shape: []const i32) @This() {
    return .{ .info = .{ .handle = cffi.mlx_array_new_data(
        @ptrCast(items.ptr),
        shape.ptr,
        @intCast(shape.len),
        @intFromEnum(DType.of(T)),
    ) } };
}

pub fn deinit(self: @This()) void {
    _ = cffi.mlx_array_free(self.info.handle);
}

/// Rebind this handle to src's value (mlx_array_set).
pub fn assign(self: *@This(), src: @This()) Error!void {
    try check(cffi.mlx_array_set(&self.info.handle, src.info.handle));
}

/// New owned handle to the same underlying array — a refcount bump
/// (mlx_array_new + mlx_array_set; mlx-c has no explicit retain), NOT a
/// data copy. For a graph-level copy of the data, use the mlx.copy op.
pub fn clone(self: @This()) Error!@This() {
    var h = cffi.mlx_array_new();
    check(cffi.mlx_array_set(&h, self.info.handle)) catch |err| {
        _ = cffi.mlx_array_free(h);
        return err;
    };
    return .{ .info = .{ .handle = h } };
}

pub fn eval(self: @This()) Error!void {
    try check(cffi.mlx_array_eval(self.info.handle));
}

/// Evaluates and copies the single element out.
pub fn item(self: @This(), comptime T: type) Error!T {
    var v: T = undefined;
    try check(itemFn(T)(&v, self.info.handle));
    return v;
}

/// View of the evaluated array's buffer; call eval() first.
/// Valid while the array is alive and unmodified.
pub fn data(self: @This(), comptime T: type) Error![]const T {
    const p = dataFn(T)(self.info.handle);
    if (p == null) return error.Mlx;
    return p[0..cffi.mlx_array_size(self.info.handle)];
}

pub fn format(self: @This(), w: *stdz.Io.Writer) stdz.Io.Writer.Error!void {
    var s = cffi.mlx_string_new();
    defer _ = cffi.mlx_string_free(s);
    if (cffi.mlx_array_tostring(&s, self.info.handle) != 0) return w.writeAll("<invalid mlx.Array>");
    try w.writeAll(stdz.mem.span(cffi.mlx_string_data(s)));
}

fn itemFn(comptime T: type) fn ([*c]T, cffi.mlx_array) callconv(.c) c_int {
    return switch (T) {
        bool => cffi.mlx_array_item_bool,
        u8 => cffi.mlx_array_item_uint8,
        u16 => cffi.mlx_array_item_uint16,
        u32 => cffi.mlx_array_item_uint32,
        u64 => cffi.mlx_array_item_uint64,
        i8 => cffi.mlx_array_item_int8,
        i16 => cffi.mlx_array_item_int16,
        i32 => cffi.mlx_array_item_int32,
        i64 => cffi.mlx_array_item_int64,
        f16 => cffi.mlx_array_item_float16,
        f32 => cffi.mlx_array_item_float32,
        f64 => cffi.mlx_array_item_float64,
        else => @compileError("no MLX item accessor for " ++ @typeName(T)),
    };
}

fn dataFn(comptime T: type) fn (cffi.mlx_array) callconv(.c) [*c]const T {
    return switch (T) {
        bool => cffi.mlx_array_data_bool,
        u8 => cffi.mlx_array_data_uint8,
        u16 => cffi.mlx_array_data_uint16,
        u32 => cffi.mlx_array_data_uint32,
        u64 => cffi.mlx_array_data_uint64,
        i8 => cffi.mlx_array_data_int8,
        i16 => cffi.mlx_array_data_int16,
        i32 => cffi.mlx_array_data_int32,
        i64 => cffi.mlx_array_data_int64,
        f16 => cffi.mlx_array_data_float16,
        f32 => cffi.mlx_array_data_float32,
        f64 => cffi.mlx_array_data_float64,
        else => @compileError("no MLX data accessor for " ++ @typeName(T)),
    };
}
