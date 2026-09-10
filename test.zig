const std = @import("std");
const mlx = @import("mlx");
const build_options = @import("build_options");

// GPU when the build has Metal, CPU otherwise: keeps `zig build test` config-agnostic.
fn stream() mlx.Error!mlx.Stream {
    return if (mlx.metalAvailable()) mlx.Stream.gpu() else mlx.Stream.cpu();
}

// Track an op result in `scope` and read it back as f32s.
fn f32s(scope: *mlx.Scope, result: mlx.Error!mlx.Array) ![]const f32 {
    return (try scope.track(result)).data(f32);
}

test "basic functionality works" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s = try stream();
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
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s = try stream();
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
    const cpu: mlx.Stream = try .cpu();
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

// Ops introduced in mlx 0.32: proves the regenerated surface (root fn, Array method, Scope
// chain) and, on Metal, the new searchsorted kernel in the metallib.
test "generated ops: new in mlx 0.32" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s = try stream();
    defer s.deinit();

    const sorted: mlx.Array = .fromSlice(f32, &.{ 1, 3, 5, 7 }, &.{4});
    defer sorted.deinit();
    const values: mlx.Array = .fromSlice(f32, &.{ 0, 3, 6, 8 }, &.{4});
    defer values.deinit();
    const idx = try mlx.searchsorted(sorted, values, "left", s);
    defer idx.deinit();
    try idx.eval();
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3, 4 }, try idx.data(u32));

    const m: mlx.Array = .fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer m.deinit();
    const tr = try m.trace(s);
    defer tr.deinit();
    try std.testing.expectEqual(5.0, try tr.item(f32));

    // flip is a negative-stride view: data() refuses it (the buffer is not laid out like the
    // shape says) and strides() shows why; contiguous() materializes a readable copy.
    const view = try m.flip(s);
    defer view.deinit();
    try std.testing.expectError(error.NotContiguous, view.data(f32));
    try std.testing.expectEqualSlices(i64, &.{ -2, -1 }, view.info.strides());
    try std.testing.expectError(error.DType, m.data(i32));
    var scope: mlx.Scope = .init(std.testing.allocator);
    defer scope.deinit();
    const flipped = try scope.enter(try view.clone(), s).contiguous(false).collect();
    defer flipped.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 3, 2, 1 }, try flipped.data(f32));
    try std.testing.expectEqualSlices(i64, &.{ 2, 1 }, flipped.info.strides());

    const cpu: mlx.Stream = try .cpu();
    defer cpu.deinit();
    const det = try mlx.linalg.det(m, cpu);
    defer det.deinit();
    try std.testing.expectApproxEqAbs(-2.0, try det.item(f32), 1e-5);
}

test "fromSlice, shape, data" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s = try stream();
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
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red

    // y = x^n where n is decided at runtime by item() reads mid-trace:
    // multiply until y >= 100. For x=2: y=2^7=128, dy/dx = 7*2^6 = 448.
    const f = struct {
        fn pow_until(inputs: mlx.Arrays) mlx.Error!mlx.Arrays {
            const s: mlx.Stream = try .cpu();
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
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
    defer s.deinit();

    // f(x) = x*x; f'(x) = 2x
    const f = struct {
        fn square(inputs: mlx.Arrays) mlx.Error!mlx.Arrays {
            const st: mlx.Stream = try .cpu();
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

// -(x*x + x): three elementwise ops, enough for compile() to fuse them into one Compiled
// primitive. Only that reaches a runtime kernel build (g++ on CPU, a Metal kernel from the
// embedded base preambles on GPU); a single op never fuses, so `square` above never gets there.
fn Fused(comptime pick: fn () mlx.Error!mlx.Stream) type {
    return struct {
        fn f(inputs: mlx.Arrays) mlx.Error!mlx.Arrays {
            const s = try pick();
            defer s.deinit();
            const x = try inputs.at(0);
            defer x.deinit();
            const sq = try mlx.multiply(x, x, s);
            defer sq.deinit();
            const sum = try mlx.add(sq, x, s);
            defer sum.deinit();
            const y = try mlx.negative(sum, s);
            defer y.deinit();
            return .fromSlice(&.{y});
        }
    };
}

test "compiled closure fuses an elementwise chain through the runtime JIT" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    inline for (.{ mlx.Stream.cpu, stream }) |pick| {
        const fun: mlx.Closure = .init(Fused(pick).f);
        defer fun.deinit();
        const compiled = try mlx.compile(fun, false);
        defer compiled.deinit();

        const x: mlx.Array = .scalar(3.0);
        defer x.deinit();
        const input: mlx.Arrays = .fromSlice(&.{x});
        defer input.deinit();
        const out = try compiled.apply(input);
        defer out.deinit();
        const y = try out.at(0);
        defer y.deinit();

        // Without -Dcpu-jit a CPU-default build skips compilation altogether (still -12), but a
        // GPU-default build fuses CPU-stream graphs too and can only refuse them at eval.
        if (pick == mlx.Stream.cpu and !build_options.cpu_jit and mlx.metalAvailable()) {
            try std.testing.expectError(error.Mlx, y.item(f32));
            try std.testing.expect(std.mem.indexOf(u8, mlx.lastError(), "CPU kernel JIT is disabled") != null);
        } else {
            try std.testing.expectEqual(-12.0, try y.item(f32));
        }
    }
}

// One op per kernel family on the default stream, so a broken metallib entry or a wrong JIT
// preamble fails here rather than in a consumer. Gather/scatter are always JIT-built from the
// base preambles, even in AOT builds.
test "kernel families: metallib groups and base JIT preambles" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s = try stream();
    defer s.deinit();
    var scope: mlx.Scope = .init(std.testing.allocator);
    defer scope.deinit();
    const t = std.testing;

    const a = try scope.track(mlx.Array.fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }));
    const b = try scope.track(mlx.Array.fromSlice(f32, &.{ 5, 6, 7, 8 }, &.{ 2, 2 }));
    const ones2 = try scope.track(mlx.ones(&.{2}, .float32, s));
    const zeros2 = try scope.track(mlx.zeros(&.{2}, .float32, s));
    const u = try scope.track(mlx.Array.fromSlice(f32, &.{ 1, 2, 3 }, &.{3}));
    const w = try scope.track(mlx.Array.fromSlice(f32, &.{ 4, 5, 6 }, &.{3}));

    // steel gemm, gemv, dot
    try t.expectEqualSlices(f32, &.{ 19, 22, 43, 50 }, try f32s(&scope, mlx.matmul(a, b, s)));
    try t.expectEqualSlices(f32, &.{ 3, 7 }, try f32s(&scope, mlx.matmul(a, ones2, s)));
    try t.expectEqualSlices(f32, &.{32}, try f32s(&scope, mlx.inner(u, w, s)));
    // reduce, arg_reduce
    try t.expectEqualSlices(f32, &.{10}, try f32s(&scope, mlx.sum(a, false, s)));
    try t.expectEqualSlices(f32, &.{4}, try f32s(&scope, mlx.max(a, false, s)));
    try t.expectEqual(3, try (try scope.track(mlx.argmax(a, false, s))).item(u32));
    // softmax, logsumexp, unary
    try t.expectEqualSlices(f32, &.{ 0.5, 0.5 }, try f32s(&scope, mlx.softmax(zeros2, false, s)));
    try t.expectApproxEqAbs(@log(2.0), (try f32s(&scope, mlx.logsumexp(zeros2, false, s)))[0], 1e-6);
    try t.expectEqualSlices(f32, &.{ 1, 1 }, try f32s(&scope, mlx.exp(zeros2, s)));
    // sort, scan, arange
    const unsorted = try scope.track(mlx.Array.fromSlice(f32, &.{ 3, 1, 2 }, &.{3}));
    try t.expectEqualSlices(f32, &.{ 1, 2, 3 }, try f32s(&scope, mlx.sort(unsorted, s)));
    try t.expectEqualSlices(u32, &.{ 1, 2, 0 }, try (try scope.track(mlx.argsort(unsorted, s))).data(u32));
    const seq = try scope.track(mlx.arange(1, 5, 1, .float32, s));
    try t.expectEqualSlices(f32, &.{ 1, 3, 6, 10 }, try f32s(&scope, mlx.cumsum(seq, false, true, null, s)));
    // gather, scatter (indexing preambles), ternary, binary_two
    const idx = try scope.track(mlx.Array.fromSlice(i32, &.{ 2, 0 }, &.{2}));
    const vals = try scope.track(mlx.Array.fromSlice(f32, &.{ 10, 20, 30 }, &.{3}));
    try t.expectEqualSlices(f32, &.{ 30, 10 }, try f32s(&scope, mlx.take(vals, idx, s)));
    const one_idx = try scope.track(mlx.Array.fromSlice(i32, &.{1}, &.{1}));
    const seven = try scope.track(mlx.Array.fromSlice(f32, &.{7}, &.{1}));
    const zeros3 = try scope.track(mlx.zeros(&.{3}, .float32, s));
    try t.expectEqualSlices(f32, &.{ 0, 7, 0 }, try f32s(&scope, mlx.putAlongAxis(zeros3, one_idx, seven, 0, s)));
    const cond = try scope.track(mlx.Array.fromSlice(bool, &.{ true, false }, &.{2}));
    try t.expectEqualSlices(f32, &.{ 1, 0 }, try f32s(&scope, mlx.where(cond, ones2, zeros2, s)));
    const quot_rem = try mlx.divmod(seven, seq, s); // 7 / [1,2,3,4]
    defer quot_rem.deinit();
    try t.expectEqualSlices(f32, &.{ 7, 3, 2, 1 }, try f32s(&scope, quot_rem.at(0)));
    try t.expectEqualSlices(f32, &.{ 0, 1, 1, 3 }, try f32s(&scope, quot_rem.at(1)));
    // conv, fft
    const signal = try scope.track(mlx.Array.fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{ 1, 4, 1 }));
    const taps = try scope.track(mlx.Array.fromSlice(f32, &.{ 1, 1 }, &.{ 1, 2, 1 }));
    try t.expectEqualSlices(f32, &.{ 3, 5, 7 }, try f32s(&scope, mlx.conv1d(signal, taps, 1, 0, 1, 1, s)));
    const impulse = try scope.track(mlx.Array.fromSlice(f32, &.{ 1, 0, 0, 0 }, &.{4}));
    const spectrum = try scope.track(mlx.fft.fft(impulse, 4, 0, .backward, s));
    try t.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, try f32s(&scope, mlx.abs(spectrum, s)));
    // quantize + quantized matmul: ones quantize exactly, so every output is K = 64
    const wide = try scope.track(mlx.ones(&.{ 8, 64 }, .float32, s));
    const row = try scope.track(mlx.ones(&.{ 1, 64 }, .float32, s));
    const parts = try mlx.quantize(wide, null, null, "affine", null, s);
    defer parts.deinit();
    const wq = try scope.track(parts.at(0));
    const scales = try scope.track(parts.at(1));
    const biases = try scope.track(parts.at(2));
    for (try f32s(&scope, mlx.quantizedMatmul(row, wq, scales, biases, true, null, null, "affine", s))) |e| {
        try t.expectApproxEqAbs(64.0, e, 1e-2);
    }
}

test "fast ops: rms_norm, layer_norm, rope, sdpa kernels and the cross_entropy fallback" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s = try stream();
    defer s.deinit();
    var scope: mlx.Scope = .init(std.testing.allocator);
    defer scope.deinit();
    const t = std.testing;

    const xs = [_]f32{ 1, 2, 3, 4 };
    const x = try scope.track(mlx.Array.fromSlice(f32, &xs, &.{ 1, 4 }));
    const w = try scope.track(mlx.ones(&.{4}, .float32, s));

    // rms_norm: x / sqrt(mean(x^2)), mean(x^2) = 7.5
    for (try f32s(&scope, mlx.fast.rmsNorm(x, w, 1e-5, s)), xs) |got, xi| try t.expectApproxEqAbs(xi / @sqrt(7.5), got, 1e-4);
    // layer_norm without affine: mean 2.5, var 1.25
    for (try f32s(&scope, mlx.fast.layerNorm(x, null, null, 0, s)), xs) |got, xi| try t.expectApproxEqAbs((xi - 2.5) / @sqrt(1.25), got, 1e-4);
    // rope at position 0 is the identity
    const x3 = try scope.track(mlx.Array.fromSlice(f32, &xs, &.{ 1, 1, 4 }));
    try t.expectEqualSlices(f32, &xs, try f32s(&scope, mlx.fast.rope(x3, 4, false, 10000, 1, 0, null, s)));
    // sdpa with a single key: attention weight 1, output is v (head dim 64 takes the fused kernel)
    const qkv = try scope.track(mlx.ones(&.{ 1, 1, 1, 64 }, .float32, s));
    for (try f32s(&scope, mlx.fast.scaledDotProductAttention(qkv, qkv, qkv, 1.0, "", null, null, false, s))) |e| try t.expectEqual(1.0, e);
    // cross_entropy of uniform logits is ln(2)
    const logits = try scope.track(mlx.zeros(&.{ 1, 2 }, .float32, s));
    const target = try scope.track(mlx.Array.fromSlice(i32, &.{0}, &.{1}));
    try t.expectApproxEqAbs(@log(2.0), (try f32s(&scope, mlx.fast.crossEntropy(logits, target, s)))[0], 1e-6);
}

test "Stream constructors surface backend failures; getMetallibPath" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    var buf: [1024]u8 = undefined;
    if (mlx.metalAvailable()) {
        const s: mlx.Stream = try .gpu();
        s.deinit();
        try std.testing.expectEqualStrings("", try mlx.getMetallibPath(&buf)); // no override installed
    } else {
        try std.testing.expectError(error.Mlx, mlx.Stream.gpu());
        try std.testing.expect(std.mem.indexOf(u8, mlx.lastError(), "gpu") != null);
        try std.testing.expectError(error.Mlx, mlx.Stream.init(.gpu));
        try std.testing.expectError(error.Mlx, mlx.getMetallibPath(&buf));
    }
    const fresh: mlx.Stream = try .init(.cpu);
    defer fresh.deinit();
    try fresh.synchronize();
}

// mlx 0.32 made CPU streams thread-affine like GPU ones: the command encoder is registered on
// the creating thread. threadUnsafe() registers globally instead.
test "streams are thread-affine; Stream.threadUnsafe is not" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const affine: mlx.Stream = try .cpu();
    defer affine.deinit();
    const shared: mlx.Stream = try .threadUnsafe(.cpu);
    defer shared.deinit();

    const Worker = struct {
        affine: mlx.Stream,
        shared: mlx.Stream,
        result: anyerror!void = {},

        fn run(self: *@This()) void {
            self.result = self.body();
        }

        fn body(self: *@This()) !void {
            const x: mlx.Array = .scalar(2.0);
            defer x.deinit();
            const y = try mlx.add(x, x, self.affine); // graph building is fine ...
            defer y.deinit();
            try std.testing.expectError(error.Mlx, y.eval()); // ... evaluating is not
            try std.testing.expect(std.mem.indexOf(u8, mlx.lastError(), "current thread") != null);

            const z = try mlx.add(x, x, self.shared);
            defer z.deinit();
            try std.testing.expectEqual(4.0, try z.item(f32));
        }
    };
    var worker: Worker = .{ .affine = affine, .shared = shared };
    const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    thread.join();
    try worker.result;
}

test "Array methods: ops without a Scope" {
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
    mlx.init();
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
    errdefer std.debug.print("mlx: {s}\n", .{mlx.lastError()}); // say why when a test goes red
    const s: mlx.Stream = try .cpu();
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
