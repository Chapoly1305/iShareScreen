// Experimental EDSR dual-output INT8 convolution. Not a production backend.
// CUTLASS supplies the implicit GEMM; the epilogue preserves CHW32, FP16
// residual state, and nearest-even INT8 quantization without an intermediate.
#include <cuda_runtime.h>
#include <cstdint>
#include <new>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

namespace iss {
using Half = cutlass::half_t;
using Layout = cutlass::layout::TensorNCxHWx<32>;
using Filter = cutlass::layout::TensorCxRSKx<32>;

struct DualOp {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  static int const kCount = 8;
  struct Params {
    Half const *skip;
    float const *scale, *bias, *factor;
    Half *output;
    int8_t *quant;
    int pixels;
  };
  Params p;
  CUTLASS_DEVICE DualOp(Params const &params) : p(params) {}
  CUTLASS_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_DEVICE void set_k_partition(int, int) {}
};

// Retain the standard iterator contract for the convolution kernel while
// exposing its starting coordinates to our two-output epilogue.
template<class Base> struct LocatedIterator : Base {
  using ThreadMap = typename Base::ThreadMap;
  cutlass::MatrixCoord start;
  CUTLASS_DEVICE LocatedIterator(typename Base::Params const &params,
      typename Base::Element *pointer, typename Base::TensorCoord extent,
      int thread, cutlass::MatrixCoord block)
    : Base(params, pointer, extent, thread, block),
      start(ThreadMap::initial_offset(thread) + block) {}
};

template<class Base> struct DualEpilogue : Base {
  using OutputTileIterator = LocatedIterator<typename Base::OutputTileIterator>;
  using Map = typename OutputTileIterator::ThreadMap;
  using AccumulatorFragmentIterator = typename Base::AccumulatorFragmentIterator;
  using AccumulatorTile = typename Base::AccumulatorTile;
  using OutputOp = DualOp;
  using SharedStorage = typename Base::SharedStorage;
  CUTLASS_DEVICE DualEpilogue(SharedStorage &storage, int thread, int warp, int lane)
    : Base(storage, thread, warp, lane) {}

  CUTLASS_DEVICE void operator()(DualOp const &op, OutputTileIterator dst,
      AccumulatorTile const &accumulators, OutputTileIterator) {
    AccumulatorFragmentIterator iterator(accumulators);
    static_assert(AccumulatorFragmentIterator::Fragment::kElements == 8,
                  "Epilogue requires contiguous groups of eight channels");
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename AccumulatorFragmentIterator::Fragment acc;
      iterator.load(acc);
      ++iterator;
      int pixel = dst.start.row() + (iter % Map::Iterations::kRow) * Map::Delta::kRow;
      int channel = dst.start.column() + (iter / Map::Iterations::kRow) * Map::Delta::kColumn;
      bool valid = pixel < op.p.pixels && channel < 64;
      int offset = (channel / 32 * op.p.pixels + pixel) * 32 + channel % 32;
      cutlass::AlignedArray<Half, 8> skip, out;
      cutlass::AlignedArray<int8_t, 8> quant;
      skip.clear();
      cutlass::arch::global_load<decltype(skip), sizeof(skip)>(skip, op.p.skip + offset, valid);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        // Explicit operations match the reference's disabled FMA contraction.
        float value = __fmul_rn(float(acc[i]), op.p.scale[(channel + i) % 64]);
        value = __fadd_rn(value, op.p.bias[(channel + i) % 64]);
        out[i] = Half(__fadd_rn(value, float(skip[i])));
        float rounded = nearbyintf(__fmul_rn(float(out[i]), op.p.factor[(channel + i) % 64]));
        quant[i] = int8_t(fminf(127.f, fmaxf(-128.f, rounded)));
      }
      cutlass::arch::global_store<decltype(out), sizeof(out)>(out, op.p.output + offset, valid);
      cutlass::arch::global_store<decltype(quant), sizeof(quant)>(quant, op.p.quant + offset, valid);
    }
  }
};

struct Context {
  virtual ~Context() = default;
  virtual int run(cudaStream_t stream) = 0;
  virtual int attributes(int *regs, int *shared) = 0;
};

template<int M, int WM, int WN, int Stages> struct Instance : Context {
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Definition = cutlass::conv::kernel::DefaultConv2dFprop<
      int8_t, Layout, int8_t, Filter, int8_t, Layout, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<M, 64, 64>,
      cutlass::gemm::GemmShape<WM, WN, 64>,
      cutlass::gemm::GemmShape<16, 8, 32>, DualOp, Swizzle, Stages,
      cutlass::arch::OpMultiplyAddSaturate,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kUnity>;
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
      typename Definition::Mma, DualEpilogue<typename Definition::Epilogue>,
      Swizzle, cutlass::conv::Operator::kFprop>;
  using Device = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
  Device device;

  int initialize(int h, int w, int8_t const *x, int8_t const *weight,
                 DualOp::Params params) {
    cutlass::Tensor4DCoord extent(1, h, w, 64), filter(64, 3, 3, 64);
    cutlass::conv::Conv2dProblemSize problem(extent, filter,
        {1, 1, 1, 1}, {1, 1}, {1, 1}, extent,
        cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Device::Arguments args(problem,
        {const_cast<int8_t *>(x), Layout::packed(extent)},
        {const_cast<int8_t *>(weight), Filter::packed(filter)},
        {params.quant, Layout::packed(extent)},
        {params.quant, Layout::packed(extent)}, params);
    auto status = device.can_implement(args);
    if (status != cutlass::Status::kSuccess) return 100 + int(status);
    if (device.get_workspace_size(args) != 0) return 201;
    return int(device.initialize(args));
  }
  int run(cudaStream_t stream) override { return int(device.run(stream)); }
  int attributes(int *regs, int *shared) override {
    cudaFuncAttributes attrs{};
    auto status = cudaFuncGetAttributes(&attrs, cutlass::Kernel<Kernel>);
    *regs = attrs.numRegs;
    *shared = sizeof(typename Kernel::SharedStorage);
    return int(status);
  }
};

template<class T> int create(Context **out, int h, int w, int8_t const *x,
                            int8_t const *weight, DualOp::Params params) {
  auto *instance = new(std::nothrow) T;
  if (!instance) return 202;
  int status = instance->initialize(h, w, x, weight, params);
  if (status) { delete instance; return status; }
  *out = instance;
  return 0;
}
}  // namespace iss

extern "C" int iss_cutlass_create(void **context, int config, int h, int w,
    void const *x, void const *weight, void const *skip, void const *scale,
    void const *bias, void const *factor, void *output, void *quant) {
  if (!context || h < 1 || w < 1 || int64_t(h) * w > 1280 * 720 ||
      !x || !weight || !skip || !scale || !bias || !factor || !output || !quant)
    return 203;
  *context = nullptr;
  iss::DualOp::Params params{static_cast<iss::Half const *>(skip),
      static_cast<float const *>(scale), static_cast<float const *>(bias),
      static_cast<float const *>(factor), static_cast<iss::Half *>(output),
      static_cast<int8_t *>(quant), h * w};
#define ISS_CASE(ID, M, WM, WN, S) case ID: return iss::create<iss::Instance<M, WM, WN, S>>(reinterpret_cast<iss::Context **>(context), h, w, static_cast<int8_t const *>(x), static_cast<int8_t const *>(weight), params)
  switch (config) {
    ISS_CASE(0, 64, 32, 32, 3);
    ISS_CASE(1, 128, 64, 32, 3);
    ISS_CASE(2, 64, 32, 32, 4);
    ISS_CASE(3, 128, 64, 32, 4);
    ISS_CASE(4, 128, 32, 64, 3);
    ISS_CASE(5, 64, 32, 32, 2);
    ISS_CASE(6, 256, 64, 32, 3);
    ISS_CASE(7, 256, 64, 32, 4);
    ISS_CASE(8, 128, 64, 64, 3);
    // Configuration 9 (one warp) cannot satisfy CUTLASS's interleaved
    // activation/filter iterator mapping; its failed build is preserved.
    ISS_CASE(10, 128, 64, 64, 2);
    ISS_CASE(11, 64, 32, 32, 5);
    default: return 204;
  }
#undef ISS_CASE
}
extern "C" int iss_cutlass_run(void *context, void *stream) {
  return context ? static_cast<iss::Context *>(context)->run(static_cast<cudaStream_t>(stream)) : 203;
}
extern "C" int iss_cutlass_attributes(void *context, int *regs, int *shared) {
  return context && regs && shared ? static_cast<iss::Context *>(context)->attributes(regs, shared) : 203;
}
extern "C" void iss_cutlass_destroy(void *context) {
  delete static_cast<iss::Context *>(context);
}
