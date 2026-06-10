const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const darwin = target.result.os.tag == .macos;

    const metal = b.option(bool, "metal", "Metal GPU backend (CPU backend is always built)") orelse false;
    const cuda = b.option(bool, "cuda", "CUDA GPU backend (reserved, not wired up yet)") orelse false;
    const jit = b.option(bool, "jit", "JIT Metal kernels at runtime (smaller metallib)") orelse false;
    const nax = b.option(bool, "nax", "Precompile NAX kernel variants (needs Metal toolchain >= 4.0)") orelse false;
    const ring = b.option(bool, "ring", "Distributed backend over TCP sockets") orelse false;
    const jaccl = b.option(bool, "jaccl", "Distributed backend over Thunderbolt RDMA (needs macOS >= 26.2)") orelse false;
    if (metal and target.result.os.tag != .macos) @panic("-Dmetal needs a macOS target");
    if (cuda) @panic("-Dcuda is not wired up yet");
    if (ring and target.result.os.tag == .windows) @panic("-Dring needs a POSIX target");
    if (jaccl and target.result.os.tag != .macos) @panic("-Djaccl needs a macOS target");

    const tests = b.step("test", "Run test suite");
    tests.dependOn(b.getInstallStep());

    const fmt = b.dependency("fmt", .{});
    const mlx = b.dependency("mlx", .{});
    const mlxc = b.dependency("mlxc", .{});

    const command = mlx.path("mlx/backend/cpu/make_compiled_preamble.sh");
    const codegen = b.addSystemCommand(&.{command.getPath(b)});
    const preamble = codegen.addOutputFileArg("compiled_preamble.cpp");
    codegen.addArg("clang");
    codegen.addDirectoryArg(mlx.path(""));
    codegen.addArgs(&.{ "TRUE", if (target.result.cpu.arch == .x86_64) "x86_64" else "arm64" });

    // MARK: LIBMLX
    const libmlx = b.addLibrary(.{
        .name = "mlx",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .sanitize_c = .off, // NOTE: upstream builds without UB sanitizer
        }),
    });
    libmlx.root_module.addIncludePath(mlx.path(""));
    libmlx.root_module.addIncludePath(fmt.path("include"));
    libmlx.root_module.addCMacro("MLX_VERSION", "\"0.31.2\"");
    libmlx.root_module.addCMacro("MLX_STATIC", "");
    if (darwin) {
        libmlx.root_module.addCMacro("MLX_USE_ACCELERATE", "");
        libmlx.root_module.addCMacro("ACCELERATE_NEW_LAPACK", "");
    }
    libmlx.root_module.addCMacro("FMT_HEADER_ONLY", "");
    if (metal) libmlx.root_module.addCMacro(
        "METAL_PATH",
        b.fmt("\"{s}\"", .{b.getInstallPath(.bin, "mlx.metallib")}),
    );

    // NOTE: In JIT mode NAX is always built and gated at runtime by `is_nax_available()`
    if (metal and !jit and !nax) libmlx.root_module.addCMacro("MLX_METAL_NO_NAX", "");

    if (metal) if (b.lazyDependency("metal-cpp", .{})) |metalcpp| {
        libmlx.root_module.addIncludePath(metalcpp.path(""));
    };

    if (!darwin) if (b.lazyDependency("lapack", .{})) |lapack| {
        // cblas.h/lapack.h include mangling headers that upstream generates with
        // configure_file; the .in templates have no substitutions, so a copy suffices.
        const mangling = b.addWriteFiles();
        _ = mangling.addCopyFile(lapack.path("CBLAS/include/cblas_mangling_with_flags.h.in"), "cblas_mangling.h");
        _ = mangling.addCopyFile(lapack.path("LAPACKE/include/lapacke_mangling_with_flags.h.in"), "lapacke_mangling.h");
        libmlx.root_module.addIncludePath(mangling.getDirectory());
        libmlx.root_module.addIncludePath(lapack.path("CBLAS/include"));
        libmlx.root_module.addIncludePath(lapack.path("LAPACKE/include"));
    };

    if (jaccl) libmlx.root_module.addIncludePath(mlx.path("mlx/distributed/jaccl/lib"));
    if (ring or jaccl) if (b.lazyDependency("json", .{})) |json| {
        libmlx.root_module.addIncludePath(json.path("single_include/nlohmann"));
    };

    appendCpp(b, libmlx.root_module, mlx, "mlx", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/common", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/cpu", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, if (metal) "mlx/backend/gpu" else "mlx/backend/no_gpu", &.{}) catch @panic("append");
    if (metal) appendCpp(b, libmlx.root_module, mlx, "mlx/backend/metal", &.{ "no_metal.cpp", if (jit) "nojit_kernels.cpp" else "jit_kernels.cpp" }) catch @panic("append");
    if (!metal) libmlx.root_module.addCSourceFile(.{
        .file = mlx.path("mlx/backend/metal/no_metal.cpp"),
        .flags = cxxflags,
        .language = .cpp,
    });
    appendCpp(b, libmlx.root_module, mlx, "mlx/distributed", &.{}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/distributed/ring", &.{if (ring) "no_ring.cpp" else "ring.cpp"}) catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/distributed/jaccl", &.{if (jaccl) "no_jaccl.cpp" else "jaccl.cpp"}) catch @panic("append");
    if (jaccl) appendCpp(b, libmlx.root_module, mlx, "mlx/distributed/jaccl/lib/jaccl", &.{}) catch @panic("append");
    libmlx.root_module.addCSourceFiles(.{
        .root = mlx.path(""),
        .files = &mlx_sources,
        .flags = cxxflags,
        .language = .cpp,
    });
    libmlx.root_module.addCSourceFiles(.{
        .root = mlx.path(""),
        .files = if (darwin) &[_][]const u8{
            "mlx/backend/cpu/gemms/bnns.cpp",
        } else &[_][]const u8{
            "mlx/backend/cpu/gemms/simd_fp16.cpp",
            "mlx/backend/cpu/gemms/simd_bf16.cpp",
        },
        .flags = cxxflags,
        .language = .cpp,
    });
    libmlx.root_module.addCSourceFile(.{
        .file = preamble,
        .flags = cxxflags,
        .language = .cpp,
    });
    if (metal) for (preambles) |name| libmlx.root_module.addCSourceFile(.{
        .file = embed(b, mlx, name),
        .flags = cxxflags,
        .language = .cpp,
    });
    if (metal and jit) for (jit_preambles) |name| libmlx.root_module.addCSourceFile(.{
        .file = embed(b, mlx, name),
        .flags = cxxflags,
        .language = .cpp,
    });
    if (darwin) libmlx.root_module.linkFramework("Accelerate", .{});
    if (metal) {
        libmlx.root_module.linkFramework("Metal", .{});
        libmlx.root_module.linkFramework("Foundation", .{});
        libmlx.root_module.linkFramework("QuartzCore", .{});
    }
    libmlx.root_module.link_libcpp = true;
    b.installArtifact(libmlx);

    // MARK: METAL KERNELS
    if (metal) {
        const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal" });
        for (kernels) |k| link.addFileArg(air(b, mlx, k));
        if (!jit) for (nojit_kernels) |k| link.addFileArg(air(b, mlx, k));
        if (!jit and nax) for (nax_kernels) |k| link.addFileArg(air(b, mlx, k));
        link.addArg("-o");
        const metallib = link.addOutputFileArg("mlx.metallib");
        const install = b.addInstallBinFile(metallib, "mlx.metallib");
        b.addNamedLazyPath("metallib", metallib);
        b.getInstallStep().dependOn(&install.step);
    }

    // MARK: LIBMLXC
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

    // MARK: MLX MODULE
    const ffi = blk: {
        const tc = b.addTranslateC(.{
            .target = target,
            .optimize = optimize,
            .root_source_file = mlxc.path("mlx/c/mlx.h"),
        });
        tc.addIncludePath(mlxc.path(""));
        const mod = tc.createModule();
        mod.linkLibrary(libmlxc);
        mod.link_libcpp = true;
        break :blk mod;
    };
    const gen = b.addExecutable(.{
        .name = "gen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("gen.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const gen_run = b.addRunArtifact(gen);
    const gen_out = gen_run.addOutputFileArg("mlx.zig");
    for ([_][]const u8{ "ops.h", "linalg.h", "fft.h", "random.h" }) |h| {
        gen_run.addFileArg(mlxc.path(b.fmt("mlx/c/{s}", .{h})));
    }
    const wrapper_root = b.addWriteFiles();
    _ = wrapper_root.addCopyFile(gen_out, "mlx.zig");
    _ = wrapper_root.addCopyFile(b.path("core.zig"), "core.zig");
    _ = wrapper_root.addCopyFile(b.path("transforms.zig"), "transforms.zig");

    const wrapper = b.addModule("mlx", .{
        .target = target,
        .optimize = optimize,
        .root_source_file = wrapper_root.getDirectory().path(b, "mlx.zig"),
        .imports = &.{.{ .name = "c", .module = ffi }},
    });

    tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("test.zig"),
        .imports = &.{.{ .name = "mlx", .module = wrapper }},
    }) })).step);
}

// MARK: C++ SOURCES

// The -U keeps __ARM_FEATURE_BF16 off in every C++ TU: clang predefines it on bf16-capable hosts
// (M2+), flipping mlx's half_types.h to native __bf16, which lacks the cross-type conversions
// array.h needs. libmlx and libmlxc must agree — the choice is ABI-visible in mangled names.
// Only the preprocessor branch is pinned; codegen may still emit bf16 instructions.
const cxxflags = &[_][]const u8{ "-std=c++20", "-U__ARM_FEATURE_BF16" };

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
            .flags = cxxflags,
            .language = .cpp,
        });
    }
}

// Preprocess a metal kernel header into a C++ preamble string (mlx::core::metal::<name>()).
fn embed(b: *std.Build, dep: *std.Build.Dependency, name: []const u8) std.Build.LazyPath {
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
    "mlx/backend/cpu/gemms/cblas.cpp",

    "mlx/backend/cuda/no_cuda.cpp",

    "mlx/distributed/mpi/no_mpi.cpp",
    "mlx/distributed/nccl/no_nccl.cpp",

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

// Extra preambles embedded in JIT mode: kernels are runtime-compiled from these instead of the metallib.
const jit_preambles = [_][]const u8{
    "arange",                                  "copy",                                     "unary",                                    "binary",                                      "binary_two",
    "fft",                                     "logsumexp",                                "ternary",                                  "softmax",                                     "scan",
    "sort",                                    "reduce",                                   "quantized_utils",                          "quantized",                                   "fp_quantized",
    "gemv_masked",                             "quantized_nax",                            "fp_quantized_nax",                         "steel/gemm/gemm",                             "steel/gemm/gemm_nax",
    "steel/gemm/kernels/steel_gemm_fused",     "steel/gemm/kernels/steel_gemm_masked",     "steel/gemm/kernels/steel_gemm_gather",     "steel/gemm/kernels/steel_gemm_splitk",        "steel/gemm/kernels/steel_gemm_segmented",
    "steel/gemm/kernels/steel_gemm_fused_nax", "steel/gemm/kernels/steel_gemm_gather_nax", "steel/gemm/kernels/steel_gemm_splitk_nax", "steel/gemm/kernels/steel_gemm_segmented_nax", "steel/conv/conv",
    "steel/conv/kernels/steel_conv",           "steel/conv/kernels/steel_conv_3d",         "steel/conv/kernels/steel_conv_general",    "steel/attn/kernels/steel_attention",          "steel/attn/kernels/steel_attention_nax",
};

// Metallib kernels precompiled in every mode (fence needs metal>=320).
const kernels = [_][]const u8{
    "arg_reduce", "conv", "gemv",                         "layer_norm", "random",
    "rms_norm",   "rope", "scaled_dot_product_attention", "fence",
};

// Additional kernels precompiled when not in JIT mode.
const nojit_kernels = [_][]const u8{
    "arange",                               "binary",                               "binary_two",                              "copy",                                "fft",
    "reduce",                               "quantized",                            "fp_quantized",                            "scan",                                "softmax",
    "logsumexp",                            "sort",                                 "ternary",                                 "unary",                               "gemv_masked",
    "steel/conv/kernels/steel_conv",        "steel/conv/kernels/steel_conv_3d",     "steel/conv/kernels/steel_conv_general",   "steel/gemm/kernels/steel_gemm_fused", "steel/gemm/kernels/steel_gemm_gather",
    "steel/gemm/kernels/steel_gemm_masked", "steel/gemm/kernels/steel_gemm_splitk", "steel/gemm/kernels/steel_gemm_segmented", "steel/attn/kernels/steel_attention",
};

// NAX (neural accelerator) kernel variants, runtime-gated by is_nax_available().
const nax_kernels = [_][]const u8{
    "quantized_nax",
    "fp_quantized_nax",
    "steel/gemm/kernels/steel_gemm_fused_nax",
    "steel/gemm/kernels/steel_gemm_gather_nax",
    "steel/gemm/kernels/steel_gemm_splitk_nax",
    "steel/gemm/kernels/steel_gemm_segmented_nax",
    "steel/attn/kernels/steel_attention_nax",
};
