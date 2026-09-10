# MLX for Zig
These are ergonomic binding's to Apple's MLX machine-learning framework. The
entire package as well as all dependencies are built with the Zig compiler,
leveraging the Zig build system.

One of the biggest benefits of this is that we don't need to link `libc` or
`libcpp` at all, and don't need to link anything else for CPU-only builds. As
for GPU-enabled builds, we only link against the minimal set of truly necessary
system libraries and frameworks, which on Apple platforms boils down to
`libobjc`, `Foundation`, `Accelerate`, `Metal`, etc.

> [!WARNING]
> **This has only been tested (both building and running) on macOS so far.**
>
> Upstream MLX supports Linux and NVIDIA hardware. I hope to support them in
> the future, but this is completely untested as of right now.


## Usage
```zig
const mlx = b.dependency("mlx", .{
    .target = target,
    .optimize = optimize,
    .metal = true,   // this is only supported on macOS.
    .nax = true,     // whether to also precompile the NAX kernels (needs Metal 4 + macOS SDK >= 26.2).
    .jit = false,    // whether to JIT compile the Metal kernels, or compile them ahead of time.
    .jaccl = false,  // enables the `jaccl` distributed backend (needs macOS >= 26.2)
    .ring = false,   // enables the `ring` distributed backend
});

// needed only if the `.metal` backend is enabled (or use `mlx.setMetallibPath` at runtime)
b.getInstallStep().dependOn(&b.addInstallBinFile(mlx.namedLazyPath("metallib"), "mlx.metallib").step);

// MLX Zig module (raw C FFI available under `mlx.cffi`)
const mlx = mlx.module("mlx");
```


## Future
- Allow user to supply their own kernels;
- Investigate how possible it would be to have `jaccl` work on Linux targets by
  swapping Apple's custom `librdma.dylib` for the upstream `rdma-core` where it
  was apparently ported from and should have a very similar interface to;

