#pragma once

//
// tessera-coreml-telemetry.h
//
// IOReport runtime-telemetry SCAFFOLD for the CoreML backend (design doc
// section 6). This is the conversion-tool-side scaffold: it defines the sample
// shape, the lifecycle, and the per-session battery attribution (decision C4),
// backed by a deterministic MOCK data source.
//
// Real IOReport access requires the private IOKit framework
// (/usr/lib/libIOReport.dylib) and is only linkable in a macOS/iOS-framework
// build. The channel names and the real API path are documented in
// tessera-coreml-telemetry.cpp; swapping the mock generator for the real client
// is isolated to ts_coreml_telemetry_sample().
//

#include <cstdint>
#include <string>

struct ts_coreml_telemetry_config {
    bool        enable;                 // master switch
    int         sampling_interval_ms;   // target cadence (design: 10ms = 100 Hz)
    std::string output_path;            // summary/sidecar path (empty = none)
};

// One telemetry sample. Power fields are milliwatts; matches sample_t in design
// section 6.2 (dram added per the channel table; battery per section 6.3). The
// _t suffix disambiguates the type from the ts_coreml_telemetry_sample() method
// (C++ shares the struct/function namespace).
struct ts_coreml_telemetry_sample_t {
    int64_t timestamp_ns;
    double  ane_power_mw;
    double  gpu_power_mw;
    double  cpu_power_mw;
    double  dram_power_mw;
    double  ane_activity_pct;           // 0..100, 0 if the DVFS channel is absent
    int     thermal_state;              // 0=nominal 1=fair 2=serious 3=critical
    int     battery_current_ma;         // signed; negative = discharging
};

// Per-session accumulated energy (decision C4: per-session for v1). mW summed
// over time -> mJ. The hero metric mWh/token is a query over these rows.
struct ts_coreml_telemetry_session_energy {
    double  ane_mj;
    double  gpu_mj;
    double  cpu_mj;
    double  dram_mj;
    double  total_mj;
    int64_t n_samples;
    double  duration_s;
};

struct ts_coreml_telemetry {
    ts_coreml_telemetry_config config;
    bool     running;
    uint64_t rng_state;                 // mock LCG state (deterministic)
    int64_t  start_ns;
    int64_t  last_ns;
    ts_coreml_telemetry_session_energy energy;
};

ts_coreml_telemetry_config ts_coreml_telemetry_default_config();

// Lifecycle. start seeds the mock source and resets the session accumulators.
// Returns 0 on success, -1 on error (err_msg set if non-null).
int  ts_coreml_telemetry_start(ts_coreml_telemetry * t,
                               const ts_coreml_telemetry_config * config,
                               std::string * err_msg);
void ts_coreml_telemetry_stop(ts_coreml_telemetry * t);

// Draw one sample from the (mock) source and fold its energy into the session
// accumulator. Returns 0 on success, -1 if not running.
int  ts_coreml_telemetry_sample(ts_coreml_telemetry * t,
                                ts_coreml_telemetry_sample_t * out);

// Snapshot the per-session energy attribution.
void ts_coreml_telemetry_session_energy_get(const ts_coreml_telemetry * t,
                                            ts_coreml_telemetry_session_energy * out);

// Write a JSON summary (config + session energy) to `path`. This is the
// "report telemetry config" pipeline step. Returns 0 on success.
int  ts_coreml_telemetry_write_summary(const ts_coreml_telemetry * t,
                                       const char * path,
                                       std::string * err_msg);
