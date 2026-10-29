#ifndef TF_LITE_STATIC_MEMORY
#error "main.cpp must be compiled with -DTF_LITE_STATIC_MEMORY (see platformio.ini / src/CMakeLists.txt)"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const unsigned char ultra_can_model[] __attribute__((aligned(16)));
extern const unsigned int ultra_can_model_len;
extern const unsigned char ultra_can_model_conv1d[] __attribute__((aligned(16)));
extern const unsigned int ultra_can_model_conv1d_len;

namespace {

constexpr int kArenaBytes = 192 * 1024;
alignas(16) uint8_t g_arena[kArenaBytes];

constexpr int kWarmup = 5;
constexpr int kRuns = 20;

class UsProfiler : public tflite::MicroProfilerInterface {
 public:
    uint32_t BeginEvent(const char *tag) override {
        int idx = -1;
        for (int i = 0; i < count_; ++i) {
            if (tags_[i] == tag) {
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            if (count_ >= kMaxTags) return 0;
            idx = count_++;
            tags_[idx] = tag;
            total_us_[idx] = 0;
            calls_[idx] = 0;
        }
        cur_ = idx;
        start_us_ = esp_timer_get_time();
        return static_cast<uint32_t>(idx);
    }

    void EndEvent(uint32_t) override {
        if (cur_ < 0) return;
        total_us_[cur_] += esp_timer_get_time() - start_us_;
        ++calls_[cur_];
        cur_ = -1;
    }

    void Reset() {
        count_ = 0;
        cur_ = -1;
    }

    void Dump(const char *label) {
        printf("PROFILE %s tags=%d\n", label, count_);
        for (int i = 0; i < count_; ++i) {
            printf("OP %-18s calls=%d total_us=%lld avg_us=%.1f\n", tags_[i],
                   calls_[i], static_cast<long long>(total_us_[i]),
                   calls_[i] ? static_cast<double>(total_us_[i]) / calls_[i] : 0.0);
        }
    }

 private:
    static constexpr int kMaxTags = 24;
    const char *tags_[kMaxTags] = {};
    int64_t total_us_[kMaxTags] = {};
    int calls_[kMaxTags] = {};
    int count_ = 0;
    int cur_ = -1;
    int64_t start_us_ = 0;
};

bool add_ops(tflite::MicroMutableOpResolver<16> &resolver) {
#define UC_ADD(expr)                                  \
    do {                                              \
        const TfLiteStatus st = (expr);               \
        if (st != kTfLiteOk) {                        \
            printf("RESOLVER_FAIL %s st=%d\n", #expr, \
                   static_cast<int>(st));             \
            return false;                             \
        }                                             \
    } while (0)

    UC_ADD(resolver.AddReshape());
    UC_ADD(resolver.AddConv2D());
    UC_ADD(resolver.AddFullyConnected());
    UC_ADD(resolver.AddSoftmax());
    UC_ADD(resolver.AddMul());
    UC_ADD(resolver.AddSum());
    UC_ADD(resolver.AddReduceMax());
    UC_ADD(resolver.AddConcatenation());
    UC_ADD(resolver.AddAdd());
    UC_ADD(resolver.AddTranspose());
    UC_ADD(resolver.AddBatchMatMul());
    UC_ADD(resolver.AddMean());
#undef UC_ADD
    return true;
}

void print_tensor_info(const char *tag, TfLiteTensor *t, size_t index) {
    if (t == nullptr) {
        printf("%s idx=%u ptr=NULL\n", tag, static_cast<unsigned>(index));
        return;
    }
    printf("%s idx=%u type=%d bytes=%u data=%p dims=", tag,
           static_cast<unsigned>(index), static_cast<int>(t->type),
           static_cast<unsigned>(t->bytes), static_cast<void *>(t->data.data));
    if (t->dims == nullptr) {
        printf("NULL\n");
        return;
    }
    for (int d = 0; d < t->dims->size; ++d) {
        printf("%d%s", t->dims->data[d], d + 1 == t->dims->size ? "\n" : "x");
    }
    if (t->dims->size == 0) printf("\n");
}

bool fill_inputs(tflite::MicroInterpreter &interpreter) {
    bool ok = true;
    for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
        TfLiteTensor *t = interpreter.input(i);
        print_tensor_info("INPUT", t, i);
        if (t == nullptr) return false;
        if (t->data.data == nullptr) {
            printf("INPUT_DATA_NULL idx=%u\n", static_cast<unsigned>(i));
            return false;
        }
        if (t->type == kTfLiteInt8) {
            memset(t->data.int8, 0, t->bytes);
        } else if (t->type == kTfLiteFloat32) {
            memset(t->data.f, 0, t->bytes);
        } else {
            printf("UNSUPPORTED_INPUT_TYPE idx=%u type=%d\n",
                   static_cast<unsigned>(i), static_cast<int>(t->type));
            ok = false;
        }
    }
    return ok;
}

void bench_model(const char *label, const unsigned char *model_data,
                 unsigned int model_len) {
    printf("=== %s : %u bytes flatbuffer ===\n", label, model_len);

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("BAD_SCHEMA %s version=%u\n", label,
               static_cast<unsigned>(model->version()));
        return;
    }

    tflite::MicroMutableOpResolver<16> resolver;
    if (!add_ops(resolver)) {
        printf("ABORT %s resolver\n", label);
        return;
    }

    UsProfiler profiler;
    tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaBytes,
                                         nullptr,
                                         &profiler);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ALLOCATE_FAIL %s arena=%d\n", label, kArenaBytes);
        return;
    }

    printf("MODEL %s inputs=%u outputs=%u arena_used=%u heap=%u\n", label,
           static_cast<unsigned>(interpreter.inputs_size()),
           static_cast<unsigned>(interpreter.outputs_size()),
           static_cast<unsigned>(interpreter.arena_used_bytes()),
           static_cast<unsigned>(esp_get_free_heap_size()));

    for (size_t i = 0; i < interpreter.outputs_size(); ++i) {
        print_tensor_info("OUTPUT", interpreter.output(i), i);
    }

    if (!fill_inputs(interpreter)) {
        printf("ABORT %s input fill\n", label);
        return;
    }

    for (int i = 0; i < kWarmup; ++i) {
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("INVOKE_FAIL %s warmup %d\n", label, i);
            return;
        }
    }

    profiler.Reset();
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("INVOKE_FAIL %s profile\n", label);
        return;
    }
    profiler.Dump(label);

    std::vector<int64_t> samples;
    samples.reserve(kRuns);
    for (int i = 0; i < kRuns; ++i) {
        const int64_t t0 = esp_timer_get_time();
        const TfLiteStatus status = interpreter.Invoke();
        const int64_t t1 = esp_timer_get_time();
        if (status != kTfLiteOk) {
            printf("INVOKE_FAIL %s run %d\n", label, i);
            return;
        }
        samples.push_back(t1 - t0);
        printf("RUN %s i=%d us=%lld\n", label, i,
               static_cast<long long>(t1 - t0));
    }

    std::sort(samples.begin(), samples.end());
    int64_t sum = 0;
    for (int64_t v : samples) sum += v;
    const size_t p95_idx = static_cast<size_t>(0.95 * (samples.size() - 1));

    printf("BENCH %s warmup=%d runs=%d min_us=%lld median_us=%lld p95_us=%lld "
           "max_us=%lld mean_us=%.1f arena_used=%u heap=%u\n",
           label, kWarmup, kRuns, static_cast<long long>(samples.front()),
           static_cast<long long>(samples[samples.size() / 2]),
           static_cast<long long>(samples[p95_idx]),
           static_cast<long long>(samples.back()),
           static_cast<double>(sum) / samples.size(),
           static_cast<unsigned>(interpreter.arena_used_bytes()),
           static_cast<unsigned>(esp_get_free_heap_size()));
    fflush(stdout);
}

constexpr int kN = 100;
constexpr int kLags = 33;
constexpr float kMu = 3.9f;

alignas(16) float g_xr[kN];
alignas(16) float g_yi[kN];
alignas(16) float g_yq[kN];
alignas(16) float g_ri[kN];
alignas(16) float g_rq[kN];
alignas(16) float g_prof[3 * kLags];
int g_prof_argmax = 0;
float g_prof_peak = 0.0f;

void fill_burst(float seed) {
    uint32_t lcg = 12345u;
    auto next = [&lcg]() {
        lcg = lcg * 1664525u + 1013904223u;
        return static_cast<float>(lcg >> 8) * (1.0f / 16777216.0f) - 0.5f;
    };
    float v = seed;
    float x[kN];
    for (int n = 0; n < kN; ++n) {
        v = kMu * v * (1.0f - v);
        x[n] = 2.0f * v - 1.0f;
    }
    for (int n = 0; n < kN; ++n) {
        const float echo = n >= 7 ? 0.35f * x[n - 7] : 0.0f;
        g_yi[n] = 0.80f * x[n] + echo + 0.02f * next();
        g_yq[n] = 0.30f * x[n] - 0.25f * echo + 0.02f * next();
    }
}

void gen_reference(float seed) {
    float v = seed;
    for (int n = 0; n < kN; ++n) {
        v = kMu * v * (1.0f - v);
        g_xr[n] = 2.0f * v - 1.0f;
    }
}

float sic_cancel() {
    float num_re = 0.0f, num_im = 0.0f, den = 0.0f;
    for (int n = 0; n < kN; ++n) {
        num_re += g_xr[n] * g_yi[n];
        num_im += g_xr[n] * g_yq[n];
        den += g_xr[n] * g_xr[n];
    }
    const float inv = 1.0f / den;
    const float c_re = num_re * inv;
    const float c_im = num_im * inv;
    for (int n = 0; n < kN; ++n) {
        g_ri[n] = g_yi[n] - c_re * g_xr[n];
        g_rq[n] = g_yq[n] - c_im * g_xr[n];
    }
    return c_re * c_re + c_im * c_im;
}

float delay_profile() {
    float best = -1.0f;
    int best_lag = 0;
    for (int lag = 1; lag <= kLags; ++lag) {
        float acc_re = 0.0f, acc_im = 0.0f;
        for (int n = lag; n < kN; ++n) {
            const float xv = g_xr[n - lag];
            acc_re += g_ri[n] * xv;
            acc_im += g_rq[n] * xv;
        }
        const float mag = sqrtf(acc_re * acc_re + acc_im * acc_im);
        g_prof[3 * (lag - 1) + 0] = mag;
        g_prof[3 * (lag - 1) + 1] = acc_re;
        g_prof[3 * (lag - 1) + 2] = acc_im;
        if (mag > best) {
            best = mag;
            best_lag = lag;
        }
    }
    g_prof_argmax = best_lag;
    g_prof_peak = best;
    return best + static_cast<float>(best_lag);
}

void report(const char *stage, std::vector<int64_t> &v) {
    std::sort(v.begin(), v.end());
    int64_t sum = 0;
    for (int64_t s : v) sum += s;
    const size_t p95 = static_cast<size_t>(0.95 * (v.size() - 1));
    printf("BENCH_PREP stage=%-7s runs=%u min_us=%lld median_us=%lld p95_us=%lld "
           "max_us=%lld mean_us=%.2f\n",
           stage, static_cast<unsigned>(v.size()),
           static_cast<long long>(v.front()),
           static_cast<long long>(v[v.size() / 2]),
           static_cast<long long>(v[p95]), static_cast<long long>(v.back()),
           static_cast<double>(sum) / v.size());
}

void bench_preprocessing() {
    std::vector<int64_t> ref_us, sic_us, prof_us, total_us;
    ref_us.reserve(kRuns);
    sic_us.reserve(kRuns);
    prof_us.reserve(kRuns);
    total_us.reserve(kRuns);

    volatile float sink = 0.0f;
    for (int i = 0; i < kWarmup + kRuns; ++i) {
        const float seed = 0.371f + 0.0007f * static_cast<float>(i);
        fill_burst(seed);
        const int64_t t0 = esp_timer_get_time();
        gen_reference(seed);
        const int64_t t1 = esp_timer_get_time();
        sink = sink + sic_cancel();
        const int64_t t2 = esp_timer_get_time();
        sink = sink + delay_profile();
        const int64_t t3 = esp_timer_get_time();
        if (i >= kWarmup) {
            ref_us.push_back(t1 - t0);
            sic_us.push_back(t2 - t1);
            prof_us.push_back(t3 - t2);
            total_us.push_back(t3 - t0);
        }
    }
    report("ref", ref_us);
    report("sic", sic_us);
    report("profile", prof_us);
    report("total", total_us);
    printf("PREP_CHECK profile_argmax=%d peak=%.4f\n", g_prof_argmax,
           static_cast<double>(g_prof_peak));
    printf("PREP_PROFILE p1=%.4f p2=%.4f p3=%.4f\n",
           static_cast<double>(g_prof[0]), static_cast<double>(g_prof[1]),
           static_cast<double>(g_prof[2]));
    fflush(stdout);
}

}

extern "C" void app_main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const esp_err_t wdt = esp_task_wdt_deinit();
    printf("BENCH_EARLY wdt_deinit=%d\n", static_cast<int>(wdt));

    esp_chip_info_t chip;
    esp_chip_info(&chip);
#ifdef CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ
    const int cpu_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#else
    const int cpu_mhz = -1;
#endif
    printf("BENCH_BOOT cores=%d rev=%d cpu_mhz=%d heap=%u psram=%u\n", chip.cores,
           chip.revision, cpu_mhz,
           static_cast<unsigned>(esp_get_free_heap_size()),
           static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)));

    printf("ABI tf_tensor_bytes=%u bytes_offset=%u ptr_bytes=%u arena=%d\n",
           static_cast<unsigned>(sizeof(TfLiteTensor)),
           static_cast<unsigned>(offsetof(TfLiteTensor, bytes)),
           static_cast<unsigned>(sizeof(void *)), kArenaBytes);
    fflush(stdout);

    {
        volatile uint32_t acc = 0;
        const int64_t t0 = esp_timer_get_time();
        for (uint32_t i = 0; i < 1000000u; ++i) acc += i;
        const int64_t t1 = esp_timer_get_time();
        printf("CPU_CAL 1M_iter_us=%lld acc=%u\n",
               static_cast<long long>(t1 - t0), static_cast<unsigned>(acc));
    }

    printf("BENCH_SETTLE waiting 6 s so the serial capture can attach\n");
    fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(6000));

    bench_preprocessing();

    bench_model("qkv-int8", ultra_can_model, ultra_can_model_len);
    bench_model("conv1d-int8", ultra_can_model_conv1d,
                ultra_can_model_conv1d_len);

    printf("BENCH_DONE heap=%u\n", static_cast<unsigned>(esp_get_free_heap_size()));
    fflush(stdout);
    while (true) {
        printf("BENCH_ALIVE t=%lld\n", static_cast<long long>(esp_timer_get_time()));
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}


