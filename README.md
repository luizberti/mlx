## Future
- Allow user to supply their own kernels;
- Port kernels over to Zig and compile to ptx/spirv directly for portability;
- Support CUDA and NCCL in a sovereign way by porting the `ioctl` stuff from
  tinygrad over to Zig, this way we don't need to link against NVIDIA's stuff;
- Investigate how possible it would be to have `jaccl` work on Linux targets by
  swapping Apple's custom `librdma.dylib` for the upstream `rdma-core` where it
  was apparently ported from and should have a very similar interface to;
