const std = @import("std");
const mlx = @import("mlx");

test "basic functionality works" {
    mlx.init();
    const s: mlx.Stream = if (mlx.metalAvailable()) .gpu() else .cpu();
    defer s.deinit();

    const a: mlx.Array = .scalar(2.0);
    defer a.deinit();
    const b: mlx.Array = .scalar(3.0);
    defer b.deinit();

    var res: mlx.Array = .{ .h = mlx.c.mlx_array_new() };
    defer res.deinit();
    try mlx.check(mlx.c.mlx_add(&res.h, a.h, b.h, s.h));

    try std.testing.expectEqual(5.0, try res.item(f32));
    try std.testing.expectEqual(mlx.Dtype.float32, res.dtype());
    try std.testing.expectEqual(0, res.ndim());
}

test "fromSlice, shape, data" {
    const s: mlx.Stream = if (mlx.metalAvailable()) .gpu() else .cpu();
    defer s.deinit();

    const a: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 3 }, a.shape());

    const two: mlx.Array = .scalar(2.0);
    defer two.deinit();
    var doubled: mlx.Array = .{ .h = mlx.c.mlx_array_new() };
    defer doubled.deinit();
    try mlx.check(mlx.c.mlx_multiply(&doubled.h, a.h, two.h, s.h));
    try doubled.eval();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8, 10, 12 }, try doubled.data(f32));
}

test "lastError captures failure message" {
    mlx.init();
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const a: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3 }, &.{3});
    defer a.deinit();
    const b: mlx.Array = .fromSlice(f32, &.{ 1, 2 }, &.{2});
    defer b.deinit();

    var res: mlx.Array = .{ .h = mlx.c.mlx_array_new() };
    defer res.deinit();
    try std.testing.expectError(error.Mlx, mlx.check(mlx.c.mlx_add(&res.h, a.h, b.h, s.h)));
    try std.testing.expect(std.mem.indexOf(u8, mlx.lastError(), "broadcast") != null);
}
