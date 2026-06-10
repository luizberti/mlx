## Usage
```zig
const mal = b.dependency("mal", .{
    .target = target,
    .optimize = optimize,
    .metal = true,   // this is only supported on macOS.
    .nax = true,     // whether to also precompile the NAX kernels (needs Metal toolchain >= 4.0).
    .jit = false,    // whether to JIT compile the Metal kernels, or compile them ahead of time.
    .jaccl = false,  // enables the `jaccl` distributed backend (needs macOS >= 26.2)
    .ring = false,   // enables the `ring` distributed backend
});

// needed only if the `.metal` backend is enabled
b.getInstallStep().dependOn(&b.addInstallBinFile(mal.namedLazyPath("metallib"), "mlx.metallib").step);

// MLX Zig module (raw C FFI available under `mlx.cffi`)
const mlx = mal.module("mlx");
```


## Future
- Allow user to supply their own kernels;
- Port kernels over to Zig and compile to ptx/spirv directly for portability;
- Support CUDA and NCCL in a sovereign way by porting the `ioctl` stuff from
  tinygrad over to Zig, this way we don't need to link against NVIDIA's stuff;
- Investigate how possible it would be to have `jaccl` work on Linux targets by
  swapping Apple's custom `librdma.dylib` for the upstream `rdma-core` where it
  was apparently ported from and should have a very similar interface to;
