//! # Scope
//! Ownership arena + op chain. Frees every tracked Array at deinit, so long op
//! chains need no per-step defer — freeing intermediates is always safe because
//! downstream graph nodes hold their own references to the arrays they consume.
//! At build time gen.zig appends one chain method per eligible op (Array-first,
//! single Array out, stream-last): each threads `result` through and no-ops once
//! it holds an error; collect() surfaces the value or the first failure.

const stdz = @import("std");
const core = @import("core.zig");
const root = @import("root.zig");
const Error = core.Error;
const Array = core.Array;
const Arrays = core.Arrays;
const DType = core.DType;
const Stream = core.Stream;
const cffi = core.c;

pub const ChainError = Error || stdz.mem.Allocator.Error;

alloc: stdz.mem.Allocator,
arena: stdz.ArrayList(cffi.mlx_array) = .empty,

stream: Stream = undefined,
result: ChainError!Array = undefined,

pub fn init(alloc: stdz.mem.Allocator) @This() {
    return .{ .alloc = alloc };
}

pub fn deinit(self: *@This()) void {
    for (self.arena.items) |h| _ = cffi.mlx_array_free(h);
    self.arena.deinit(self.alloc);
}

/// Reset the scope and begin a new chain on `stream`: frees everything tracked
/// so far (keeping the arena's capacity), then seeds result with `operand`,
/// taking ownership of it. To keep using the array after the scope frees it,
/// pass a new reference instead: `scope.enter(try x.clone(), s)`.
pub fn enter(self: *@This(), operand: Array, stream: Stream) *@This() {
    for (self.arena.items) |h| _ = cffi.mlx_array_free(h);
    self.arena.clearRetainingCapacity();

    self.result = self.track(operand);
    self.stream = stream;
    return self;
}

/// Surface the chain result as a NEW reference (refcount bumped via
/// mlx_array_new + mlx_array_set — mlx-c has no explicit retain). The caller
/// owns it and it survives deinit; the scope still frees its own references.
pub fn collect(self: *@This()) ChainError!Array {
    const arr = try self.result;
    var h = cffi.mlx_array_new();
    core.check(cffi.mlx_array_set(&h, arr.info.handle)) catch |err| {
        _ = cffi.mlx_array_free(h);
        return err;
    };
    return .{ .info = .{ .handle = h } };
}

/// Takes an op result directly, so a manual chain step is a single try:
/// `const t = try scope.track(mlx.add(a, b, s));`
pub fn track(self: *@This(), array: Error!Array) (Error || stdz.mem.Allocator.Error)!Array {
    const arr = try array;
    self.arena.append(self.alloc, arr.info.handle) catch |err| {
        arr.deinit();
        return err;
    };
    return arr;
}

/// Untrack `arr` and hand ownership back to the caller (it survives deinit).
pub fn escape(self: *@This(), arr: Array) Array {
    for (self.arena.items, 0..) |h, i| {
        if (h.ctx == arr.info.handle.ctx) {
            _ = self.arena.swapRemove(i);
            break;
        }
    }
    return arr;
}
