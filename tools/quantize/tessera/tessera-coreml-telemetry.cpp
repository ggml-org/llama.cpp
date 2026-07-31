#include "tessera-coreml-telemetry.h"

#include <chrono>
#include <fstream>

//
// Real IOReport path (for the macOS/iOS-framework-linked build).
//
// IOReport is a private framework at /usr/lib/libIOReport.dylib. The client
// flow (mirrors `powermetrics` and the design doc section 6.1/6.2):
//
//   1. void * conn = IOReportCopyAllChannels(kIOReportCategoryPower, ...);
//      // or IOReportCopyChannelsInGroup("Energy Model", ...)
//   2. void * a = IOReportCreateSamples(conn);   // first shot
//      ... wait sampling_interval ...
//      void * b = IOReportCreateSamples(conn);   // second shot
//   3. void * delta = IOReportCreateSamplesDelta(a, b, ...);
//      // per-channel energy delta in mJ over the interval; P(mW) = E(mJ)/dt(s)
//
// Channels (design 6.1), group "Energy Model" unless noted:
//   "ANE0"/"ANE1"  ANE power (mJ), summed over both ANE instances
//   "GPU Power"    GPU power (mJ)
//   "CPU Power"    CPU package power, E + P cores (mJ)
//   "DRAM Power"   DRAM power (mJ)
//   "ANE Activity" (group "ANE DVFS") power-normalized ANE activity estimate (%)
//
// Battery (design 6.3): IOKit service "AppleSmartBattery", property
//   "InstantAmperage" (mA, signed; negative = discharge).
// Thermal (design 6.4): public NSProcessInfo.thermalState (0..3).
//
// None of that is linkable here (private framework + no Foundation in the
// portable build), so ts_coreml_telemetry_sample() below returns deterministic
// plausible values from an LCG. Only that one function changes when this moves
// to a framework-linked build; the sample struct, lifecycle, and per-session
// attribution are already the real shape.
//

static int64_t ts_tel_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// SplitMix64-style step; deterministic given the seed so tests are repeatable.
static uint64_t ts_tel_next(ts_coreml_telemetry * t) {
    t->rng_state += 0x9e3779b97f4a7c15ull;
    uint64_t z = t->rng_state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Uniform double in [lo, hi].
static double ts_tel_rand(ts_coreml_telemetry * t, double lo, double hi) {
    const double u = (double) (ts_tel_next(t) >> 11) * (1.0 / 9007199254740992.0);
    return lo + u * (hi - lo);
}

ts_coreml_telemetry_config ts_coreml_telemetry_default_config() {
    ts_coreml_telemetry_config c;
    c.enable               = false;
    c.sampling_interval_ms = 10;    // 100 Hz (design 6.2)
    c.output_path          = "";
    return c;
}

int ts_coreml_telemetry_start(ts_coreml_telemetry * t,
                              const ts_coreml_telemetry_config * config,
                              std::string * err_msg) {
    if (t == nullptr) {
        if (err_msg) {
            *err_msg = "telemetry handle is null";
        }
        return -1;
    }
    t->config    = config ? *config : ts_coreml_telemetry_default_config();
    t->running   = true;
    t->rng_state = 0x123456789abcdef0ull;   // fixed seed -> reproducible mock
    t->start_ns  = ts_tel_now_ns();
    t->last_ns   = t->start_ns;
    t->energy    = {};
    return 0;
}

void ts_coreml_telemetry_stop(ts_coreml_telemetry * t) {
    if (t) {
        t->running = false;
    }
}

int ts_coreml_telemetry_sample(ts_coreml_telemetry * t,
                               ts_coreml_telemetry_sample_t * out) {
    if (t == nullptr || out == nullptr || !t->running) {
        return -1;
    }

    const int64_t now = ts_tel_now_ns();
    double dt_s = (double) (now - t->last_ns) / 1e9;
    if (dt_s <= 0.0) {
        // first sample (or clock granularity): assume the configured cadence
        dt_s = (double) t->config.sampling_interval_ms / 1000.0;
    }
    t->last_ns = now;

    // --- mock source: plausible iPhone-under-inference power rails ---
    out->timestamp_ns     = now;
    out->ane_power_mw     = ts_tel_rand(t, 800.0, 1600.0);   // ANE busy
    out->gpu_power_mw     = ts_tel_rand(t, 200.0, 500.0);
    out->cpu_power_mw     = ts_tel_rand(t, 600.0, 1200.0);
    out->dram_power_mw    = ts_tel_rand(t, 300.0, 500.0);
    out->ane_activity_pct = ts_tel_rand(t, 40.0, 95.0);
    out->thermal_state    = (int) ts_tel_rand(t, 0.0, 3.0);  // nominal..critical

    // battery current from total power at a nominal 3.8 V cell (mA, discharge)
    const double total_mw = out->ane_power_mw + out->gpu_power_mw +
                            out->cpu_power_mw + out->dram_power_mw;
    out->battery_current_ma = -(int) (total_mw / 3.8);

    // --- per-session energy attribution (C4): mW * s = mJ ---
    t->energy.ane_mj  += out->ane_power_mw  * dt_s;
    t->energy.gpu_mj  += out->gpu_power_mw  * dt_s;
    t->energy.cpu_mj  += out->cpu_power_mw  * dt_s;
    t->energy.dram_mj += out->dram_power_mw * dt_s;
    t->energy.total_mj = t->energy.ane_mj + t->energy.gpu_mj +
                         t->energy.cpu_mj + t->energy.dram_mj;
    t->energy.n_samples++;
    t->energy.duration_s = (double) (now - t->start_ns) / 1e9;

    return 0;
}

void ts_coreml_telemetry_session_energy_get(const ts_coreml_telemetry * t,
                                            ts_coreml_telemetry_session_energy * out) {
    if (t && out) {
        *out = t->energy;
    }
}

int ts_coreml_telemetry_write_summary(const ts_coreml_telemetry * t,
                                      const char * path,
                                      std::string * err_msg) {
    if (t == nullptr || path == nullptr) {
        if (err_msg) {
            *err_msg = "null argument";
        }
        return -1;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        if (err_msg) {
            *err_msg = std::string("cannot open ") + path;
        }
        return -1;
    }
    const auto & e = t->energy;
    f << "{\n";
    f << "  \"schema\": \"tessera.coreml.telemetry.v1\",\n";
    f << "  \"source\": \"mock\",\n";
    f << "  \"config\": {\n";
    f << "    \"enable\": " << (t->config.enable ? "true" : "false") << ",\n";
    f << "    \"sampling_interval_ms\": " << t->config.sampling_interval_ms << ",\n";
    f << "    \"output_path\": \"" << t->config.output_path << "\"\n";
    f << "  },\n";
    f << "  \"session_energy\": {\n";
    f << "    \"ane_mj\": "  << e.ane_mj  << ",\n";
    f << "    \"gpu_mj\": "  << e.gpu_mj  << ",\n";
    f << "    \"cpu_mj\": "  << e.cpu_mj  << ",\n";
    f << "    \"dram_mj\": " << e.dram_mj << ",\n";
    f << "    \"total_mj\": " << e.total_mj << ",\n";
    f << "    \"n_samples\": " << e.n_samples << ",\n";
    f << "    \"duration_s\": " << e.duration_s << "\n";
    f << "  }\n";
    f << "}\n";
    if (!f.good()) {
        if (err_msg) {
            *err_msg = std::string("write failed for ") + path;
        }
        return -1;
    }
    return 0;
}
