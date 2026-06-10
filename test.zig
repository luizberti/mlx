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
    try std.testing.expectEqual(mlx.DType.float32, res.info.dtype());
    try std.testing.expectEqual(0, res.info.ndim());
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
    const key = try mlx.random.prng(42);
    defer key.deinit();
    const zero: mlx.Array = .scalar(0.0);
    defer zero.deinit();
    const one: mlx.Array = .scalar(1.0);
    defer one.deinit();
    const r = try mlx.random.uniform(zero, one, &.{ 2, 2 }, .float32, key, s);
    defer r.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 2 }, r.info.shape());

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
    try std.testing.expectEqualSlices(i32, &.{ 2, 3 }, a.info.shape());

    const two: mlx.Array = .scalar(2.0);
    defer two.deinit();
    var doubled: mlx.Array = .{ .info = .{ .handle = mlx.cffi.mlx_array_new() } };
    defer doubled.deinit();
    try mlx.check(mlx.cffi.mlx_multiply(&doubled.info.handle, a.info.handle, two.info.handle, s.h));
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

test "Array methods: ops without a Scope" {
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const x: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{4});
    defer x.deinit();

    const doubled = try x.add(x, s);
    defer doubled.deinit();
    const total = try doubled.sum(false, s);
    defer total.deinit();
    try std.testing.expectEqual(20.0, try total.item(f32));

    const grid = try x.reshape(&.{ 2, 2 }, s);
    defer grid.deinit();
    try std.testing.expectEqualSlices(i32, &.{ 2, 2 }, grid.info.shape());

    // vector-out op as a method
    const halves = try grid.split(2, 0, s);
    defer halves.deinit();
    try std.testing.expectEqual(2, halves.len());
}

test "Scope tracks intermediates, escape survives deinit" {
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const a: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{4});
    defer a.deinit();

    var scope: mlx.Scope = .init(std.testing.allocator);
    const doubled = try scope.track(mlx.add(a, a, s)); // {2,4,6,8}
    const squared = try scope.track(mlx.multiply(doubled, doubled, s)); // {4,16,36,64}
    const total = try scope.track(mlx.sum(squared, false, s)); // 120
    const out = scope.escape(total);
    defer out.deinit();
    scope.deinit();

    try std.testing.expectEqual(120.0, try out.item(f32));
}

test "Scope chain: sticky result, take survives scope deinit" {
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    // no defer: enter() takes ownership, the scope frees x
    const x: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{4});

    var scope: mlx.Scope = .init(std.testing.allocator);
    const out = try scope.enter(x, s).add(x).square().sum(false).collect();
    defer out.deinit();
    scope.deinit(); // collect() bumped the refcount: out must outlive the scope

    try std.testing.expectEqual(120.0, try out.item(f32)); // sum((2x)^2) = 4+16+36+64
}

test "Scope.track frees the array when arena append fails" {
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const a: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{4});
    defer a.deinit();
    const y = try mlx.add(a, a, s); // no defer: ownership goes to track, even on failure
    try y.eval(); // materialize y's buffer so a leak would show in active memory

    var before: usize = 0;
    try mlx.check(mlx.cffi.mlx_get_active_memory(&before));

    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var scope: mlx.Scope = .init(failing.allocator());
    defer scope.deinit();
    try std.testing.expectError(error.OutOfMemory, scope.track(y));

    var after: usize = 0;
    try mlx.check(mlx.cffi.mlx_get_active_memory(&after));
    try std.testing.expect(after < before); // y's buffer was released, not leaked
}

test "Scope chain: enter resets the scope for reuse" {
    mlx.init();
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const x: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{4});
    defer x.deinit();
    const bad: mlx.Array = .fromSlice(f32, &.{ 1, 2 }, &.{2});
    defer bad.deinit();

    var scope: mlx.Scope = .init(std.testing.allocator);
    defer scope.deinit();

    // clone: each enter consumes a reference, the caller's x stays valid
    const first = try scope.enter(try x.clone(), s).add(x).sum(false).collect();
    defer first.deinit();
    try std.testing.expectEqual(20.0, try first.item(f32));

    // poison the second chain, then re-enter: result and arena must reset
    try std.testing.expectError(error.Mlx, scope.enter(try x.clone(), s).add(bad).collect());
    const third = try scope.enter(try x.clone(), s).square().sum(false).collect();
    defer third.deinit();
    try std.testing.expectEqual(30.0, try third.item(f32)); // 1+4+9+16

    // first chain's collected result is an independent reference: still valid
    try std.testing.expectEqual(20.0, try first.item(f32));
}

test "Scope chain: failure poisons, take surfaces it" {
    mlx.init();
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const x: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3 }, &.{3}); // owned by the scope
    const bad: mlx.Array = .fromSlice(f32, &.{ 1, 2 }, &.{2});
    defer bad.deinit();

    var scope: mlx.Scope = .init(std.testing.allocator);
    defer scope.deinit();
    // add fails (broadcast), square no-ops on the poisoned result
    try std.testing.expectError(error.Mlx, scope.enter(x, s).add(bad).square().collect());
    try std.testing.expect(std.mem.indexOf(u8, mlx.lastError(), "broadcast") != null);
}

test "lastError captures failure message" {
    mlx.init();
    const s: mlx.Stream = .cpu();
    defer s.deinit();

    const a: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3 }, &.{3});
    defer a.deinit();
    const b: mlx.Array = .fromSlice(f32, &.{ 1, 2 }, &.{2});
    defer b.deinit();

    var res: mlx.Array = .{ .info = .{ .handle = mlx.cffi.mlx_array_new() } };
    defer res.deinit();
    try std.testing.expectError(error.Mlx, mlx.check(mlx.cffi.mlx_add(&res.info.handle, a.info.handle, b.info.handle, s.h)));
    try std.testing.expect(std.mem.indexOf(u8, mlx.lastError(), "broadcast") != null);
}
