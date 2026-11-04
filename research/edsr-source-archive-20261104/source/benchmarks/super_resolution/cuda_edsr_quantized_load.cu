// Experimental load-time quantization for EDSR conv1. Same quantizer math,
// no persistent INT8 input buffer. This is a component feasibility test.
// Reuse the validated CHW32 output coordinates and opaque context interface.
#include "cuda_edsr_cutlass.cu"

namespace issq {
using iss::Half;
using iss::Layout;
using iss::Filter;

struct ReluOp {
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  static int const kCount = 8;
  struct Params {
    float const *scale, *bias, *factor;
    int8_t *output;
    int pixels;
  };
  Params p;
  CUTLASS_DEVICE ReluOp(Params const &params) : p(params) {}
  CUTLASS_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_DEVICE void set_k_partition(int, int) {}
};

template<class Base> struct ReluEpilogue : Base {
  using OutputTileIterator = iss::LocatedIterator<typename Base::OutputTileIterator>;
  using Map = typename OutputTileIterator::ThreadMap;
  using AccumulatorFragmentIterator = typename Base::AccumulatorFragmentIterator;
  using AccumulatorTile = typename Base::AccumulatorTile;
  using OutputOp = ReluOp;
  using SharedStorage = typename Base::SharedStorage;
  CUTLASS_DEVICE ReluEpilogue(SharedStorage &storage, int thread, int warp, int lane)
      : Base(storage, thread, warp, lane) {}
  CUTLASS_DEVICE void operator()(ReluOp const &op, OutputTileIterator dst,
      AccumulatorTile const &accumulators, OutputTileIterator) {
    AccumulatorFragmentIterator iterator(accumulators);
    static_assert(AccumulatorFragmentIterator::Fragment::kElements == 8);
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename AccumulatorFragmentIterator::Fragment acc;
      iterator.load(acc);
      ++iterator;
      int pixel = dst.start.row() + (iter % Map::Iterations::kRow) * Map::Delta::kRow;
      int channel = dst.start.column() + (iter / Map::Iterations::kRow) * Map::Delta::kColumn;
      bool valid = pixel < op.p.pixels && channel < 64;
      int offset = (channel / 32 * op.p.pixels + pixel) * 32 + channel % 32;
      cutlass::AlignedArray<int8_t, 8> output;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) {
        float value = __fadd_rn(__fmul_rn(float(acc[i]), op.p.scale[(channel + i) % 64]),
                               op.p.bias[(channel + i) % 64]);
        value = __fmul_rn(fmaxf(0.f, value), op.p.factor[(channel + i) % 64]);
        output[i] = int8_t(fminf(127.f, fmaxf(0.f, nearbyintf(value))));
      }
      cutlass::arch::global_store<decltype(output), sizeof(output)>(output, op.p.output + offset, valid);
    }
  }
};

// CUTLASS's virtual address iterator uses one-byte element offsets. Translate
// those offsets to actual FP16 addresses in this loader, then return INT8
// fragments to the unchanged Tensor Core pipeline. Kernel-level TensorRefA is
// a virtual addressing view, not the actual dtype of the allocated input.
template<class Base> struct QuantizedLoad : Base {
  using Access = typename Base::TileAccessIterator;
  using ThreadMap = typename Base::ThreadMap;
  using Fragment = typename Base::Fragment;
  using Element = typename Base::Element;
  using ConvProblemSize = typename Base::ConvProblemSize;
  struct Params : Base::Params {
    float const *factor = nullptr;
    cutlass::FastDivmod plane;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE Params(typename Base::Params const &p, int pixels)
        : Base::Params(p), plane(pixels * 32) {}
  };
  Access access;
  int8_t const *virtual_base;
  Params const &params;
  CUTLASS_HOST_DEVICE QuantizedLoad(Params const &p, ConvProblemSize const &problem,
      Element const *pointer, int thread, cutlass::MatrixCoord const &offset = {})
      : Base(p, problem, pointer, thread, offset), access(p, problem, pointer, thread, offset),
        virtual_base(pointer), params(p) {}
  CUTLASS_HOST_DEVICE static Params getParams(ConvProblemSize const &problem, Layout const &layout) {
    return Params(Base::getParams(problem, layout), problem.H * problem.W);
  }
  CUTLASS_DEVICE void load(Fragment &fragment) {
    access.set_iteration_index(0);
    fragment.clear();
    constexpr int width = Access::AccessType::kElements;
    static_assert(width == 16, "Expected 16-byte INT8 vectors");
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < ThreadMap::Iterations::kCount * Base::kAccessesPerVector; ++index) {
      if (access.valid()) {
        int offset = int(reinterpret_cast<int8_t const *>(access.get()) - virtual_base);
        int group = params.plane.div(offset);
        int channel = group * 32 + (offset & 31);
        cutlass::AlignedArray<Half, 8> values[2];
        auto pointer = reinterpret_cast<Half const *>(virtual_base) + offset;
        cutlass::arch::global_load<cutlass::AlignedArray<Half, 8>, 16>(values[0], pointer, true);
        cutlass::arch::global_load<cutlass::AlignedArray<Half, 8>, 16>(values[1], pointer + 8, true);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < width; ++i) {
          float value = __fmul_rn(float(values[i / 8][i % 8]), params.factor[channel + i]);
          fragment[index * width + i] = int8_t(fminf(127.f, fmaxf(-128.f, nearbyintf(value))));
        }
      }
      ++access;
    }
  }
  CUTLASS_HOST_DEVICE QuantizedLoad &operator++() { access.advance(); return *this; }
};

template<int M, int WM, int WN, bool Fused> struct Instance : iss::Context {
  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  using Definition = cutlass::conv::kernel::DefaultConv2dFprop<
      int8_t, Layout, int8_t, Filter, int8_t, Layout, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<M, 64, 64>, cutlass::gemm::GemmShape<WM, WN, 64>,
      cutlass::gemm::GemmShape<16, 8, 32>, ReluOp, Swizzle, 2,
      cutlass::arch::OpMultiplyAddSaturate, cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kUnity>;
  using InputIterator = typename std::conditional<Fused,
      QuantizedLoad<typename Definition::IteratorA>, typename Definition::IteratorA>::type;
  using Mma = cutlass::conv::threadblock::ImplicitGemmPipelined<
      typename Definition::Mma::Shape, InputIterator, typename Definition::SmemIteratorA,
      typename Definition::IteratorB, typename Definition::SmemIteratorB,
      int8_t, Layout, typename Definition::MmaPolicy>;
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<Mma,
      ReluEpilogue<typename Definition::Epilogue>, Swizzle, cutlass::conv::Operator::kFprop>;
  using Device = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
  typename Kernel::Params params;
  dim3 grid;
  int initialize(int h, int w, void const *input, int8_t const *weight,
                 float const *input_factor, ReluOp::Params output) {
    cutlass::Tensor4DCoord extent(1, h, w, 64), filter(64, 3, 3, 64);
    cutlass::conv::Conv2dProblemSize problem(extent, filter,
        {1, 1, 1, 1}, {1, 1}, {1, 1}, extent, cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Device::Arguments args(problem,
        {static_cast<int8_t *>(const_cast<void *>(input)), Layout::packed(extent)},
        {const_cast<int8_t *>(weight), Filter::packed(filter)},
        {output.output, Layout::packed(extent)}, {output.output, Layout::packed(extent)}, output);
    auto status = Device::can_implement(args);
    if (status != cutlass::Status::kSuccess) return 100 + int(status);
    params = typename Kernel::Params(args);
    if constexpr(Fused) params.iterator_A.factor = input_factor;
    grid = Swizzle().get_grid_shape(params.grid_tiled_shape);
    if (sizeof(typename Kernel::SharedStorage) >= 48 * 1024)
      return int(cudaFuncSetAttribute(cutlass::Kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(typename Kernel::SharedStorage)));
    return 0;
  }
  int run(cudaStream_t stream) override {
    cutlass::Kernel<Kernel><<<grid, Device::kWarpCount * 32, sizeof(typename Kernel::SharedStorage), stream>>>(params);
    return int(cudaGetLastError());
  }
  int attributes(int *registers, int *shared) override {
    cudaFuncAttributes attrs{};
    auto code = cudaFuncGetAttributes(&attrs, cutlass::Kernel<Kernel>);
    *registers = attrs.numRegs;
    *shared = sizeof(typename Kernel::SharedStorage);
    return int(code);
  }
};

template<class T> int create(void **out, int h, int w, void const *input,
    int8_t const *weight, float const *factor, ReluOp::Params params) {
  auto *context = new(std::nothrow) T;
  if (!context) return 202;
  int status = context->initialize(h, w, input, weight, factor, params);
  if (status) { delete context; return status; }
  *out = context;
  return 0;
}
}  // namespace issq

extern "C" int iss_quantized_load_create(void **context, int config, int fused, int h, int w,
    void const *input, void const *weight, void const *input_factor, void const *scale,
    void const *bias, void const *factor, void *output) {
  if (!context || !input || !weight || !input_factor || !scale || !bias || !factor || !output ||
      h < 1 || w < 1 || int64_t(h) * w > 1280 * 720 || (fused != 0 && fused != 1)) return 203;
  *context = nullptr;
  issq::ReluOp::Params params{static_cast<float const *>(scale), static_cast<float const *>(bias),
      static_cast<float const *>(factor), static_cast<int8_t *>(output), h * w};
#define ISS_Q_CASE(ID, M, WM, WN, F) case ID: return issq::create<issq::Instance<M, WM, WN, F>>(context, h, w, input, static_cast<int8_t const *>(weight), static_cast<float const *>(input_factor), params)
  if (fused) {
    switch(config) {
      ISS_Q_CASE(0, 64, 32, 32, true);
      ISS_Q_CASE(1, 128, 64, 32, true);
      ISS_Q_CASE(2, 128, 32, 64, true);
      ISS_Q_CASE(3, 256, 64, 32, true);
      default: return 204;
    }
  }
  switch(config) {
    ISS_Q_CASE(0, 64, 32, 32, false);
    ISS_Q_CASE(1, 128, 64, 32, false);
    ISS_Q_CASE(2, 128, 32, 64, false);
    ISS_Q_CASE(3, 256, 64, 32, false);
    default: return 204;
  }
#undef ISS_Q_CASE
}
