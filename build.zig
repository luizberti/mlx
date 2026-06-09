const std = @import("std");

// Crawled at configure time: mlx core, backend/common, backend/no_gpu, mlx/c. Only the
// backend-conditional sources stay explicit (gemms pick, distributed/io stubs, no_metal/no_cuda).
const mlx_cpu_srcs = [_][]const u8{
    "mlx/backend/cpu/device_info.cpp",
    "mlx/backend/cpu/arg_reduce.cpp",
    "mlx/backend/cpu/binary.cpp",
    "mlx/backend/cpu/conv.cpp",
    "mlx/backend/cpu/copy.cpp",
    "mlx/backend/cpu/distributed.cpp",
    "mlx/backend/cpu/eig.cpp",
    "mlx/backend/cpu/eigh.cpp",
    "mlx/backend/cpu/encoder.cpp",
    "mlx/backend/cpu/fft.cpp",
    "mlx/backend/cpu/hadamard.cpp",
    "mlx/backend/cpu/matmul.cpp",
    "mlx/backend/cpu/gemms/cblas.cpp",
    "mlx/backend/cpu/gemms/bnns.cpp",
    "mlx/backend/cpu/masked_mm.cpp",
    "mlx/backend/cpu/primitives.cpp",
    "mlx/backend/cpu/quantized.cpp",
    "mlx/backend/cpu/reduce.cpp",
    "mlx/backend/cpu/scan.cpp",
    "mlx/backend/cpu/select.cpp",
    "mlx/backend/cpu/softmax.cpp",
    "mlx/backend/cpu/logsumexp.cpp",
    "mlx/backend/cpu/sort.cpp",
    "mlx/backend/cpu/threefry.cpp",
    "mlx/backend/cpu/indexing.cpp",
    "mlx/backend/cpu/luf.cpp",
    "mlx/backend/cpu/qrf.cpp",
    "mlx/backend/cpu/svd.cpp",
    "mlx/backend/cpu/inverse.cpp",
    "mlx/backend/cpu/cholesky.cpp",
    "mlx/backend/cpu/unary.cpp",
    "mlx/backend/cpu/eval.cpp",
    "mlx/backend/cpu/compiled.cpp",
    "mlx/backend/cpu/jit_compiler.cpp",

    "mlx/distributed/primitives.cpp",
    "mlx/distributed/ops.cpp",
    "mlx/distributed/distributed.cpp",
    "mlx/distributed/utils.cpp",
    "mlx/distributed/mpi/no_mpi.cpp",
    "mlx/distributed/ring/no_ring.cpp",
    "mlx/distributed/nccl/no_nccl.cpp",
    "mlx/distributed/jaccl/no_jaccl.cpp",

    "mlx/io/load.cpp",
    "mlx/io/no_safetensors.cpp",
    "mlx/io/no_gguf.cpp",

    "mlx/backend/metal/no_metal.cpp",
    "mlx/backend/cuda/no_cuda.cpp",
};

fn strLess(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

// Non-recursive crawl of dep's `sub` dir, appending every `.cpp` (minus `skip`) as `sub/name`.
fn appendCpp(b: *std.Build, out: *std.ArrayList([]const u8), dep: *std.Build.Dependency, sub: []const u8, skip: []const []const u8) void {
    const io = b.graph.io;
    var dir = dep.builder.build_root.handle.openDir(io, sub, .{ .iterate = true }) catch @panic("open dir");
    defer dir.close(io);
    const start = out.items.len;
    var it = dir.iterate();
    next: while (it.next(io) catch @panic("iterate")) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".cpp")) continue;
        for (skip) |s| if (std.mem.eql(u8, e.name, s)) continue :next;
        out.append(b.allocator, b.fmt("{s}/{s}", .{ sub, e.name })) catch @panic("oom");
    }
    std.mem.sort([]const u8, out.items[start..], {}, strLess);
}

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{
        .default_target = .{
            // Pin pre-bf16 baseline so __ARM_FEATURE_BF16 stays off in *every* TU: upstream then uses its
            // struct bfloat16_t (native __bf16 lacks the cross-type conversions array.h needs). Global = ABI-consistent.
            .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 },
        },
    });

    const mlx = b.dependency("mlx", .{});
    const mlxc = b.dependency("mlxc", .{});
    const metal = b.dependency("metal-cpp", .{});
    const fmt = b.dependency("fmt", .{});
    _ = metal;

    const preamble = std.Build.Step.Run.create(b, "compiled_preamble");
    preamble.addFileArg(mlx.path("mlx/backend/cpu/make_compiled_preamble.sh"));
    const preamble_cpp = preamble.addOutputFileArg("compiled_preamble.cpp");
    preamble.addArg("clang");
    preamble.addDirectoryArg(mlx.path(""));
    preamble.addArgs(&.{ "TRUE", "arm64" });

    const libmlx = b.addLibrary(.{
        .name = "mlx",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    libmlx.root_module.addCMacro("MLX_STATIC", "");
    libmlx.root_module.addCMacro("MLX_USE_ACCELERATE", "");
    libmlx.root_module.addCMacro("ACCELERATE_NEW_LAPACK", "");
    libmlx.root_module.addCMacro("FMT_HEADER_ONLY", "");
    libmlx.root_module.addIncludePath(mlx.path(""));
    libmlx.root_module.addIncludePath(fmt.path("include"));
    var mlx_srcs: std.ArrayList([]const u8) = .empty;
    appendCpp(b, &mlx_srcs, mlx, "mlx", &.{"version.cpp"}); // version.cpp built separately for -DMLX_VERSION
    appendCpp(b, &mlx_srcs, mlx, "mlx/backend/common", &.{});
    appendCpp(b, &mlx_srcs, mlx, "mlx/backend/no_gpu", &.{});
    mlx_srcs.appendSlice(b.allocator, &mlx_cpu_srcs) catch @panic("oom");
    libmlx.root_module.addCSourceFiles(.{
        .root = mlx.path(""),
        .files = mlx_srcs.items,
        .flags = &.{"-std=c++20"},
        .language = .cpp,
    });
    libmlx.root_module.addCSourceFile(.{
        .file = mlx.path("mlx/version.cpp"),
        .flags = &.{ "-std=c++20", "-DMLX_VERSION=\"0.31.2\"" },
        .language = .cpp,
    });
    libmlx.root_module.addCSourceFile(.{
        .file = preamble_cpp,
        .flags = &.{"-std=c++20"},
        .language = .cpp,
    });
    libmlx.root_module.linkFramework("Accelerate", .{});
    libmlx.root_module.link_libcpp = true;
    b.installArtifact(libmlx);

    const libmlxc = b.addLibrary(.{
        .name = "mlxc",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    libmlxc.root_module.addCMacro("MLX_STATIC", "");
    libmlxc.root_module.addIncludePath(mlxc.path("")); // mlx-c headers
    libmlxc.root_module.addIncludePath(mlx.path("")); // mlx C++ headers it wraps
    var mlxc_srcs: std.ArrayList([]const u8) = .empty;
    appendCpp(b, &mlxc_srcs, mlxc, "mlx/c", &.{}); // object.cpp not present on disk
    libmlxc.root_module.addCSourceFiles(.{
        .root = mlxc.path(""),
        .files = mlxc_srcs.items,
        .flags = &.{"-std=c++20"},
        .language = .cpp,
    });
    libmlxc.root_module.linkLibrary(libmlx);
    b.installArtifact(libmlxc);

    const tc = b.addTranslateC(.{
        .target = target,
        .optimize = optimize,
        .root_source_file = mlxc.path("mlx/c/mlx.h"),
    });
    tc.addIncludePath(mlxc.path("")); // so `#include "mlx/c/..."` resolves

    const ffi = tc.createModule();
    const exe = b.addExecutable(.{
        .name = "mal",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("src/main.zig"),
            .imports = &.{.{ .name = "mlx", .module = ffi }},
        }),
    });
    exe.root_module.linkLibrary(libmlxc);
    exe.root_module.link_libcpp = true;
    b.installArtifact(exe);
}
