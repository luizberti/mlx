// Stands in for mlx/backend/cpu/jit_compiler.cpp when the build leaves the CPU runtime JIT out
// (the default; opt in with -Dcpu-jit). With available() false, compile() never fuses CPU
// graphs (mlx skips compilation outright when the default device is the CPU), so nothing shells
// out to g++ or writes kernels under $TMPDIR, and the prebuilt CPU preamble is not needed.
// The remaining entry points are only reachable from Compiled::eval_cpu, which a GPU-default
// build can still hit when it compiles a function that runs on a CPU stream.
#include <stdexcept>

#include "mlx/backend/cpu/jit_compiler.h"

namespace mlx::core {

namespace {
[[noreturn]] void disabled() {
  throw std::runtime_error(
      "[compile] CPU kernel JIT is disabled in this build (mal: -Dcpu-jit=false); "
      "compile() functions that run on a CPU stream cannot be fused.");
}
} // namespace

bool JitCompiler::available() {
  return false;
}

const std::tuple<bool, std::string, std::string>& JitCompiler::get_preamble() {
  disabled();
}

std::string JitCompiler::build_command(
    const std::filesystem::path&,
    const std::string&,
    const std::string&) {
  disabled();
}

std::string JitCompiler::exec(const std::string&) {
  disabled();
}

} // namespace mlx::core
