const std = @import("std");
const mlx = @import("mlx");

test "basic functionality works" {
    const s = mlx.mlx_default_gpu_stream_new();
    const a = mlx.mlx_array_new_float32(2.0);
    const b = mlx.mlx_array_new_float32(3.0);
    var res = mlx.mlx_array_new();

    _ = mlx.mlx_add(&res, a, b, s);
    _ = mlx.mlx_array_eval(res);

    var out: f32 = 0;
    _ = mlx.mlx_array_item_float32(&out, res);
    try std.testing.expectEqual(5.0, out);
}
