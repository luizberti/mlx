const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{
        .default_target = .{
            // Pin pre-bf16 baseline so __ARM_FEATURE_BF16 stays off in *every* TU: upstream then uses its
            // struct bfloat16_t (native __bf16 lacks the cross-type conversions array.h needs). Global = ABI-consistent.
            .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 },
        },
    });

    const tests = b.step("test", "Run test suite");

    const mlx = b.dependency("mlx", .{});
    const mlxc = b.dependency("mlxc", .{});
    const metal = b.dependency("metal-cpp", .{});
    const fmt = b.dependency("fmt", .{});

    const command = mlx.path("mlx/backend/cpu/make_compiled_preamble.sh");
    const codegen = b.addSystemCommand(&.{command.getPath(b)});
    const preamble = codegen.addOutputFileArg("compiled_preamble.cpp");
    codegen.addArg("clang");
    codegen.addDirectoryArg(mlx.path(""));
    codegen.addArgs(&.{ "TRUE", "arm64" });

    const libmlx = b.addLibrary(.{
        .name = "mlx",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            // metal-cpp's NS::SharedPtr does null->release() (a runtime no-op via objc_msgSend(nil));
            // UBSan traps the technically-UB null member-call. Upstream builds without UBSan.
            .sanitize_c = .off,
        }),
    });
    libmlx.root_module.addCMacro("MLX_STATIC", "");
    libmlx.root_module.addCMacro("MLX_USE_ACCELERATE", "");
    libmlx.root_module.addCMacro("ACCELERATE_NEW_LAPACK", "");
    libmlx.root_module.addCMacro("FMT_HEADER_ONLY", "");
    libmlx.root_module.addCMacro("MLX_VERSION", "\"0.31.2\"");
    libmlx.root_module.addCMacro("MLX_METAL_NO_NAX", "");
    libmlx.root_module.addCMacro("METAL_PATH", b.fmt("\"{s}\"", .{b.getInstallPath(.bin, "mlx.metallib")}));
    libmlx.root_module.addIncludePath(mlx.path(""));
    libmlx.root_module.addIncludePath(fmt.path("include"));
    libmlx.root_module.addIncludePath(metal.path(""));
    appendCpp(b, libmlx.root_module, mlx, "mlx", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/common", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/cpu", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/gpu", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/metal", &.{ "no_metal.cpp", "jit_kernels.cpp" }) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/distributed", &.{}) catch @panic("append");
    libmlx.root_module.addCSourceFiles(.{
        .root = mlx.path(""),
        .files = &mlx_sources,
        .flags = &.{"-std=c++20"},
        .language = .cpp,
    });
    libmlx.root_module.addCSourceFile(.{
        .file = preamble,
        .flags = &.{"-std=c++20"},
        .language = .cpp,
    });
    for (preambles) |name| libmlx.root_module.addCSourceFile(.{
        .file = jit(b, mlx, name),
        .flags = &.{"-std=c++20"},
        .language = .cpp,
    });
    libmlx.root_module.linkFramework("Accelerate", .{});
    libmlx.root_module.linkFramework("Metal", .{});
    libmlx.root_module.linkFramework("Foundation", .{});
    libmlx.root_module.linkFramework("QuartzCore", .{});
    libmlx.root_module.link_libcpp = true;
    b.installArtifact(libmlx);

    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal" });
    for (kernels) |k| link.addFileArg(air(b, mlx, k));
    link.addArg("-o");
    const metallib = link.addOutputFileArg("mlx.metallib");
    b.getInstallStep().dependOn(&b.addInstallBinFile(metallib, "mlx.metallib").step);

    const libmlxc = b.addLibrary(.{
        .name = "mlxc",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    libmlxc.root_module.addCMacro("MLX_STATIC", "");
    libmlxc.root_module.addIncludePath(mlxc.path(""));
    libmlxc.root_module.addIncludePath(mlx.path(""));
    appendCpp(b, libmlxc.root_module, mlxc, "mlx/c", &.{}) catch @panic("append");
    libmlxc.root_module.linkLibrary(libmlx);
    b.installArtifact(libmlxc);

    const ffi = blk: {
        const tc = b.addTranslateC(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = mlxc.path("mlx/c/mlx.h"),
        });
        tc.addIncludePath(mlxc.path(""));
        break :blk tc.createModule();
    };

    tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = blk: {
        const mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path("test.zig"),
            .imports = &.{.{ .name = "mlx", .module = ffi }},
            .link_libcpp = true,
        });
        mod.linkLibrary(libmlxc);
        break :blk mod;
    } })).step);
}

// Non-recursive crawl of dep's `sub` dir, adding each top-level `.cpp` to `mod` (minus `skip`).
fn appendCpp(b: *std.Build, mod: *std.Build.Module, dep: *std.Build.Dependency, sub: []const u8, skip: []const []const u8) !void {
    const io = b.graph.io;
    const br = dep.builder.build_root.handle;
    var dir = try br.openDir(io, sub, .{ .iterate = true });
    defer dir.close(io);

    var cursor = dir.iterate();
    outer: while (try cursor.next(io)) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".cpp")) continue;
        for (skip) |s| if (std.mem.eql(u8, e.name, s)) continue :outer;
        mod.addCSourceFile(.{
            .file = dep.path(b.fmt("{s}/{s}", .{ sub, e.name })),
            .flags = &.{"-std=c++20"},
            .language = .cpp,
        });
    }
}

// Preprocess a metal kernel header into a C++ preamble string (mlx::core::metal::<name>()).
fn jit(b: *std.Build, dep: *std.Build.Dependency, name: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{"bash"});
    run.addFileArg(dep.path("mlx/backend/metal/make_compiled_preamble.sh"));
    const out = run.addOutputDirectoryArg("jit");
    run.addArg("clang");
    run.addDirectoryArg(dep.path(""));
    run.addArg(name);
    return out.path(b, b.fmt("{s}.cpp", .{std.fs.path.basename(name)}));
}

// Compile one .metal kernel to a .air object for the metallib.
fn air(b: *std.Build, dep: *std.Build.Dependency, kernel: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{
        "xcrun", "-sdk",    "macosx",         "metal",                 "-x",                    "metal",
        "-Wall", "-Wextra", "-fno-fast-math", "-Wno-c++17-extensions", "-Wno-c++20-extensions", "-c",
    });
    run.addFileArg(dep.path(b.fmt("mlx/backend/metal/kernels/{s}.metal", .{kernel})));
    run.addPrefixedDirectoryArg("-I", dep.path(""));
    run.addArg("-o");
    return run.addOutputFileArg(b.fmt("{s}.air", .{std.fs.path.basename(kernel)}));
}

const mlx_sources = [_][]const u8{
    "mlx/backend/cpu/gemms/bnns.cpp",
    "mlx/backend/cpu/gemms/cblas.cpp",

    "mlx/backend/cuda/no_cuda.cpp",

    "mlx/distributed/mpi/no_mpi.cpp",
    "mlx/distributed/ring/no_ring.cpp",
    "mlx/distributed/nccl/no_nccl.cpp",
    "mlx/distributed/jaccl/no_jaccl.cpp",

    "mlx/io/load.cpp",
    "mlx/io/no_gguf.cpp",
    "mlx/io/no_safetensors.cpp",
};

// Base metal preambles (always compiled, even no-JIT: custom/compiled kernels JIT against them).
const preambles = [_][]const u8{
    "utils",                 "unary_ops",               "binary_ops",
    "ternary_ops",           "reduce_utils",            "hadamard",
    "indexing/scatter",      "indexing/masked_scatter", "indexing/gather",
    "indexing/gather_front", "indexing/gather_axis",    "indexing/scatter_axis",
};

// No-JIT metallib kernels (fence needs metal>=320; NAX skipped via MLX_METAL_NO_NAX).
const kernels = [_][]const u8{
    "arg_reduce",                           "conv",                                    "gemv",                                "layer_norm",                           "random",
    "rms_norm",                             "rope",                                    "scaled_dot_product_attention",        "fence",                                "arange",
    "binary",                               "binary_two",                              "copy",                                "fft",                                  "reduce",
    "quantized",                            "fp_quantized",                            "scan",                                "softmax",                              "logsumexp",
    "sort",                                 "ternary",                                 "unary",                               "gemv_masked",                          "steel/conv/kernels/steel_conv",
    "steel/conv/kernels/steel_conv_3d",     "steel/conv/kernels/steel_conv_general",   "steel/gemm/kernels/steel_gemm_fused", "steel/gemm/kernels/steel_gemm_gather", "steel/gemm/kernels/steel_gemm_masked",
    "steel/gemm/kernels/steel_gemm_splitk", "steel/gemm/kernels/steel_gemm_segmented", "steel/attn/kernels/steel_attention",
};
