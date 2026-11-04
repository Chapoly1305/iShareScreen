// Isolated fixed-format NVDEC experiment. No host pixel buffers or GPU copies.
// Headers are pristine NVIDIA PyNvVideoCodec 2.2.3 SDK headers; the executing
// driver is the same installed libnvcuvid as PyNvVideoCodec 2.2.4.
#include <cuda.h>
#include <nvcuvid.h>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

static thread_local std::string last_error;
static void check(CUresult rc, const char* operation) {
    if (rc != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(rc, &name);
        throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown"));
    }
}
#define CK(x) check((x), #x)
struct Frame {
    uint64_t pointer;
    uint32_t pitch, width, height, reserved;
    int64_t pts;
    uint64_t queries;
    double wait_ms, map_ms;
};
// The caller retains every graph and event through this synchronous parser
// call and until GPU completion. Entries are borrowed only during submit.
struct GraphEntry {
    uint64_t pointer;
    uint32_t pitch, reserved;
    uint64_t executable[2];
};
static_assert(sizeof(Frame) == 56, "Python Frame ABI mismatch");
static_assert(sizeof(GraphEntry) == 32, "Python GraphEntry ABI mismatch");
struct Decoder {
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUvideoparser parser = nullptr;
    CUvideodecoder decoder = nullptr;
    CUdeviceptr mapped = 0;
    bool polling = false, failed = false, complete_frames = false;
    Frame frame{};
    const GraphEntry* graphs = nullptr;
    unsigned int graph_count = 0, output_slot = 0;
    CUevent output_ready = nullptr;
    std::string error;

    explicit Decoder(uint64_t stream_value, bool use_polling) :
        stream(reinterpret_cast<CUstream>(stream_value)), polling(use_polling) {
        CK(cuCtxGetCurrent(&context));
        if (!context || !stream) throw std::runtime_error("Current CUDA context and explicit stream required");
        CUVIDPARSERPARAMS params{};
        params.CodecType = cudaVideoCodec_HEVC;
        params.ulMaxNumDecodeSurfaces = 1;
        params.ulClockRate = 1000;
        params.ulMaxDisplayDelay = 0;
        params.pUserData = this;
        params.pfnSequenceCallback = sequence;
        params.pfnDecodePicture = decode;
        params.pfnDisplayPicture = display;
        CK(cuvidCreateVideoParser(&parser, &params));
    }
    ~Decoder() {
        // The caller synchronizes its readers before explicit destruction.
        if (mapped) cuvidUnmapVideoFrame(decoder, mapped);
        if (parser) cuvidDestroyVideoParser(parser);
        if (decoder) cuvidDestroyDecoder(decoder);
    }
    template<class Fn> static int callback(void* opaque, Fn fn) noexcept {
        auto& self = *static_cast<Decoder*>(opaque);
        try { return fn(self); }
        catch (const std::exception& ex) { self.failed = true; self.error = ex.what(); return 0; }
    }
    static int CUDAAPI sequence(void* opaque, CUVIDEOFORMAT* fmt) {
        return callback(opaque, [fmt](Decoder& self) {
            if (self.decoder) throw std::runtime_error("Sequence reconfiguration is outside this experiment");
            if (fmt->codec != cudaVideoCodec_HEVC || fmt->chroma_format != cudaVideoChromaFormat_420 ||
                fmt->bit_depth_luma_minus8 || fmt->bit_depth_chroma_minus8 || !fmt->progressive_sequence ||
                fmt->coded_width != 1280 || fmt->coded_height != 720 ||
                fmt->display_area.left || fmt->display_area.top ||
                fmt->display_area.right != 1280 || fmt->display_area.bottom != 720 ||
                fmt->video_signal_description.matrix_coefficients != 1 ||
                !fmt->video_signal_description.video_full_range_flag)
                throw std::runtime_error("Expected fixed 1280x720 progressive HEVC8 BT709 full-range 420");
            CUVIDDECODECREATEINFO info{};
            info.CodecType = fmt->codec;
            info.ChromaFormat = fmt->chroma_format;
            info.OutputFormat = cudaVideoSurfaceFormat_NV12;
            info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
            info.ulNumOutputSurfaces = 2;
            info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
            info.ulNumDecodeSurfaces = fmt->min_num_decode_surfaces;
            info.ulWidth = info.ulTargetWidth = info.ulMaxWidth = fmt->coded_width;
            info.ulHeight = info.ulTargetHeight = info.ulMaxHeight = fmt->coded_height;
            CK(cuvidCreateDecoder(&self.decoder, &info));
            return static_cast<int>(info.ulNumDecodeSurfaces);
        });
    }
    static int CUDAAPI decode(void* opaque, CUVIDPICPARAMS* picture) {
        return callback(opaque, [picture](Decoder& self) {
            if (picture->field_pic_flag) throw std::runtime_error("Interlaced input rejected");
            CK(cuvidDecodePicture(self.decoder, picture));
            return 1;
        });
    }
    static int CUDAAPI display(void* opaque, CUVIDPARSERDISPINFO* picture) {
        return callback(opaque, [picture](Decoder& self) {
            if (!picture || self.mapped) throw std::runtime_error("At most one outstanding frame required");
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            uint64_t queries = 0;
            CUVIDGETDECODESTATUS status{};
            if (self.polling) {
                do {
                    CK(cuvidGetDecodeStatus(self.decoder, picture->picture_index, &status));
                    ++queries;
                    if (status.decodeStatus == cuvidDecodeStatus_Success) break;
                    if (status.decodeStatus != cuvidDecodeStatus_InProgress)
                        throw std::runtime_error("Invalid/failed decoder status before mapping");
                    if (Clock::now() - start > std::chrono::seconds(2))
                        throw std::runtime_error("Decode status polling timed out");
                    __builtin_ia32_pause();
                } while (true);
            }
            auto ready = Clock::now();
            CUVIDPROCPARAMS params{};
            params.progressive_frame = picture->progressive_frame;
            params.second_field = picture->repeat_first_field + 1;
            params.top_field_first = picture->top_field_first;
            params.unpaired_field = picture->repeat_first_field < 0;
            params.output_stream = self.stream;
            unsigned int pitch = 0;
            CK(cuvidMapVideoFrame(self.decoder, picture->picture_index, &self.mapped, &pitch, &params));
            auto mapped_at = Clock::now();
            CK(cuvidGetDecodeStatus(self.decoder, picture->picture_index, &status));
            if (status.decodeStatus != cuvidDecodeStatus_Success) throw std::runtime_error("Decode status not successful");
            self.frame = {self.mapped,pitch,1280,720,0,picture->timestamp,queries,
                std::chrono::duration<double,std::milli>(ready-start).count(),
                std::chrono::duration<double,std::milli>(mapped_at-ready).count()};
            for (unsigned int i = 0; i < self.graph_count; ++i) {
                const auto& graph = self.graphs[i];
                if (graph.pointer != self.mapped || graph.pitch != pitch) continue;
                auto executable = reinterpret_cast<CUgraphExec>(graph.executable[self.output_slot]);
                if (!executable || !self.output_ready) throw std::runtime_error("Invalid graph binding");
                CK(cuGraphLaunch(executable, self.stream));
                CK(cuEventRecord(self.output_ready, self.stream));
                self.frame.reserved = 1; // Same Frame ABI; graph already launched.
                break;
            }
            return 1;
        });
    }
    int submit(const uint8_t* bytes, size_t size, int64_t pts, Frame* output) {
        if (mapped) throw std::runtime_error("Release the previous mapped frame after GPU readers complete");
        if (failed) throw std::runtime_error(error);
        frame = {};
        CUVIDSOURCEDATAPACKET packet{};
        packet.payload = bytes;
        packet.payload_size = size;
        packet.timestamp = pts;
        // Match installed PyNv Decode's parser flags for the controlled test.
        packet.flags = CUVID_PKT_TIMESTAMP | (size ? (complete_frames ? CUVID_PKT_ENDOFPICTURE : 0) : CUVID_PKT_ENDOFSTREAM);
        CUresult rc = cuvidParseVideoData(parser, &packet);
        if (failed) throw std::runtime_error(error);
        CK(rc);
        *output = frame;
        return mapped ? 1 : 0;
    }
};
extern "C" const char* nv_error() { return last_error.c_str(); }
extern "C" int nv_check_context(void* ptr) noexcept {
    CUcontext current = nullptr;
    CUresult rc = cuCtxGetCurrent(&current);
    if (rc != CUDA_SUCCESS) { last_error = "Cannot read current CUDA context"; return -1; }
    if (!current || current != static_cast<Decoder*>(ptr)->context) {
        last_error = "Native decoder requires its creating CUDA context";
        return -2; // No parser, mapping, event or graph operation has happened.
    }
    return 0;
}
extern "C" void* nv_create(uint64_t stream, int polling) noexcept {
    try { return new Decoder(stream, polling != 0); }
    catch (const std::exception& ex) { last_error = ex.what(); return nullptr; }
}
extern "C" int nv_decode(void* ptr, uint64_t bytes, uint64_t size, int64_t pts, Frame* frame) noexcept {
    try { return static_cast<Decoder*>(ptr)->submit(reinterpret_cast<const uint8_t*>(bytes),size,pts,frame); }
    catch (const std::exception& ex) { last_error = ex.what(); return -1; }
}
extern "C" int nv_release(void* ptr) noexcept {
    try {
        auto& self = *static_cast<Decoder*>(ptr);
        if (!self.mapped) throw std::runtime_error("No mapped frame to release");
        CK(cuvidUnmapVideoFrame(self.decoder,self.mapped));
        self.mapped = 0;
        return 0;
    } catch (const std::exception& ex) { last_error = ex.what(); return -1; }
}
extern "C" void nv_destroy(void* ptr) noexcept { delete static_cast<Decoder*>(ptr); }

extern "C" int nv_set_complete_frames(void* ptr) noexcept {
    auto& self = *static_cast<Decoder*>(ptr);
    if (self.decoder) { last_error="Set framing before the first packet"; return -1; }
    self.complete_frames=true; return 0;
}

extern "C" int nv_decode_graph(void* ptr, uint64_t bytes, uint64_t size, int64_t pts,
    Frame* frame, const GraphEntry* graphs, unsigned int graph_count,
    unsigned int output_slot, uint64_t output_ready) noexcept {
    auto& self = *static_cast<Decoder*>(ptr);
    const int context_status = nv_check_context(ptr);
    if (context_status) return context_status;
    // Never retain bindings across calls: cache eviction cannot leave a native
    // dangling graph handle, and closing a decoder cannot destroy runtime state.
    struct Reset {
        Decoder& self;
        ~Reset() { self.graphs = nullptr; self.graph_count = 0; self.output_ready = nullptr; }
    } reset{self};
    try {
        if (graph_count > 2 || output_slot > 1 || !output_ready || (graph_count && !graphs))
            throw std::runtime_error("Invalid bounded graph submission");
        self.graphs = graphs;
        self.graph_count = graph_count;
        self.output_slot = output_slot;
        self.output_ready = reinterpret_cast<CUevent>(output_ready);
        return self.submit(reinterpret_cast<const uint8_t*>(bytes), size, pts, frame);
    } catch (const std::exception& ex) {
        last_error = ex.what();
        self.failed = true;
        self.error = last_error;
        return -1;
    }
}

extern "C" int nv_abort(void* ptr) noexcept {
    auto& self = *static_cast<Decoder*>(ptr);
    try {
        // Error cleanup only. Never unmap a surface with an outstanding reader.
        CK(cuStreamSynchronize(self.stream));
        if (self.mapped) {
            CK(cuvidUnmapVideoFrame(self.decoder, self.mapped));
            self.mapped = 0;
        }
        self.failed = true;
        self.error = "Decoder aborted after failed graph submission";
        return 0;
    } catch (const std::exception& ex) { last_error = ex.what(); return -1; }
}
