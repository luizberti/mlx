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

    const mlx = b.dependency("mlx", .{});
    const mlxc = b.dependency("mlxc", .{});
    const metal = b.dependency("metal-cpp", .{});
    const fmt = b.dependency("fmt", .{});
    _ = metal;

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
        }),
    });
    libmlx.root_module.addCMacro("MLX_STATIC", "");
    libmlx.root_module.addCMacro("MLX_USE_ACCELERATE", "");
    libmlx.root_module.addCMacro("ACCELERATE_NEW_LAPACK", "");
    libmlx.root_module.addCMacro("FMT_HEADER_ONLY", "");
    libmlx.root_module.addCMacro("MLX_VERSION", "\"0.31.2\"");
    libmlx.root_module.addIncludePath(mlx.path(""));
    libmlx.root_module.addIncludePath(fmt.path("include"));
    appendCpp(b, libmlx.root_module, mlx, "mlx") catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/common") catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/cpu") catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/backend/no_gpu") catch @panic("append");
    appendCpp(b, libmlx.root_module, mlx, "mlx/distributed") catch @panic("append");
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
    libmlxc.root_module.addIncludePath(mlxc.path(""));
    libmlxc.root_module.addIncludePath(mlx.path(""));
    appendCpp(b, libmlxc.root_module, mlxc, "mlx/c") catch @panic("append");
    libmlxc.root_module.linkLibrary(libmlx);
    b.installArtifact(libmlxc);

    const tc = b.addTranslateC(.{
        .target = target,
        .optimize = optimize,
        .root_source_file = mlxc.path("mlx/c/mlx.h"),
    });
    tc.addIncludePath(mlxc.path(""));

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

// Non-recursive crawl of dep's `sub` dir, adding each top-level `.cpp` to `mod`.
fn appendCpp(b: *std.Build, mod: *std.Build.Module, dep: *std.Build.Dependency, sub: []const u8) !void {
    const io = b.graph.io;
    const br = dep.builder.build_root.handle;
    var dir = try br.openDir(io, sub, .{ .iterate = true });
    defer dir.close(io);

    var cursor = dir.iterate();
    while (try cursor.next(io)) |e| {
        if (e.kind == .file and std.mem.endsWith(u8, e.name, ".cpp")) {
            mod.addCSourceFile(.{
                .file = dep.path(b.fmt("{s}/{s}", .{ sub, e.name })),
                .flags = &.{"-std=c++20"},
                .language = .cpp,
            });
        }
    }
}

const mlx_sources = [_][]const u8{
    "mlx/backend/cpu/gemms/bnns.cpp",
    "mlx/backend/cpu/gemms/cblas.cpp",

    "mlx/backend/metal/no_metal.cpp",
    "mlx/backend/cuda/no_cuda.cpp",

    "mlx/distributed/mpi/no_mpi.cpp",
    "mlx/distributed/ring/no_ring.cpp",
    "mlx/distributed/nccl/no_nccl.cpp",
    "mlx/distributed/jaccl/no_jaccl.cpp",

    "mlx/io/load.cpp",
    "mlx/io/no_gguf.cpp",
    "mlx/io/no_safetensors.cpp",
};
