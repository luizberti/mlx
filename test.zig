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

    const res = try mlx.add(a, b, s);
    defer res.deinit();

    try std.testing.expectEqual(5.0, try res.item(f32));
    try std.testing.expectEqual(mlx.Dtype.float32, res.dtype());
    try std.testing.expectEqual(0, res.ndim());
}

test "generated ops: nullable args, namespaces, multi-out" {
    const s: mlx.Stream = if (mlx.metalAvailable()) .gpu() else .cpu();
    defer s.deinit();

    // clip with nullable bounds (a_max = null)
    const x: mlx.Array = .fromSlice(f32, &.{ -2, 0.5, 9 }, &.{3});
    defer x.deinit();
    const lo: mlx.Array = .scalar(0.0);
    defer lo.deinit();
    const clipped = try mlx.clip(x, lo, null, s);
    defer clipped.deinit();
    try clipped.eval();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 9 }, try clipped.data(f32));

    // random namespace + dtype arg
    const key = try mlx.random.key(42);
    defer key.deinit();
    const zero: mlx.Array = .scalar(0.0);
    defer zero.deinit();
    const one: mlx.Array = .scalar(1.0);
    defer one.deinit();
    const r = try mlx.random.uniform(zero, one, &.{ 2, 2 }, .float32, key, s);
    defer r.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 2 }, r.shape());

    // linalg multi-out: QR of identity gives |Q[0][0]| = 1, Q[0][1] = 0
    const cpu: mlx.Stream = .cpu();
    defer cpu.deinit();
    const eye = try mlx.eye(2, 2, 0, .float32, cpu);
    defer eye.deinit();
    const q, const rr = try mlx.linalg.qr(eye, cpu);
    defer q.deinit();
    defer rr.deinit();
    try q.eval();
    const qd = try q.data(f32);
    try std.testing.expectApproxEqAbs(1.0, @abs(qd[0]), 1e-6);
    try std.testing.expectApproxEqAbs(0.0, qd[1], 1e-6);
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

test "vjp through data-dependent host loop" {
    mlx.init();

    // y = x^n where n is decided at runtime by item() reads mid-trace:
    // multiply until y >= 100. For x=2: y=2^7=128, dy/dx = 7*2^6 = 448.
    const f = struct {
        fn pow_until(inputs: mlx.Arrays) mlx.Error!mlx.Arrays {
            const s: mlx.Stream = .cpu();
            defer s.deinit();
            var y = try inputs.at(0);
            const x = try inputs.at(0);
            defer x.deinit();
            while (try y.item(f32) < 100) {
                const next = try mlx.multiply(y, x, s);
                y.deinit();
                y = next;
            }
            defer y.deinit();
            return .fromSlice(&.{y});
        }
    }.pow_until;

    const fun: mlx.Closure = .init(f);
    defer fun.deinit();

    const x: mlx.Array = .scalar(2.0);
    defer x.deinit();
    const one: mlx.Array = .scalar(1.0);
    defer one.deinit();
    const primals: mlx.Arrays = .fromSlice(&.{x});
    defer primals.deinit();
    const cotangents: mlx.Arrays = .fromSlice(&.{one});
    defer cotangents.deinit();

    const outputs, const grads = try mlx.vjp(fun, primals, cotangents);
    defer outputs.deinit();
    defer grads.deinit();

    const y = try outputs.at(0);
    defer y.deinit();
    const dx = try grads.at(0);
    defer dx.deinit();
    try std.testing.expectEqual(128.0, try y.item(f32));
    try std.testing.expectEqual(448.0, try dx.item(f32));
}

test "valueAndGrad and compiled closure" {
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    // f(x) = x*x; f'(x) = 2x
    const f = struct {
        fn square(inputs: mlx.Arrays) mlx.Error!mlx.Arrays {
            const st: mlx.Stream = .cpu();
            defer st.deinit();
            const x = try inputs.at(0);
            defer x.deinit();
            const y = try mlx.multiply(x, x, st);
            defer y.deinit();
            return .fromSlice(&.{y});
        }
    }.square;

    const fun: mlx.Closure = .init(f);
    defer fun.deinit();

    const vg = try mlx.valueAndGrad(fun, &.{0});
    defer vg.deinit();
    const x: mlx.Array = .scalar(3.0);
    defer x.deinit();
    const input: mlx.Arrays = .fromSlice(&.{x});
    defer input.deinit();

    const values, const grads = try vg.apply(input);
    defer values.deinit();
    defer grads.deinit();
    const v = try values.at(0);
    defer v.deinit();
    const g = try grads.at(0);
    defer g.deinit();
    try std.testing.expectEqual(9.0, try v.item(f32));
    try std.testing.expectEqual(6.0, try g.item(f32));

    const compiled = try mlx.compile(fun, false);
    defer compiled.deinit();
    const out = try compiled.apply(input);
    defer out.deinit();
    const co = try out.at(0);
    defer co.deinit();
    try std.testing.expectEqual(9.0, try co.item(f32));
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
