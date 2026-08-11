#include "ggml-trace.h"
#include "ggml-impl.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct trace_transfer {
    std::string phase;
    std::string src_name;
    std::string dst_name;
    std::string name;
    int64_t     ne[4];
    size_t      nbytes;
};

struct trace_rpc {
    std::string endpoint;
    int         device;
    std::string cmd;
    std::string name;
    int64_t     ne[4];
    size_t      bytes_out;
    size_t      bytes_in;
    int         allreduce; // index into step.allreduces, -1 if outside one
};

struct trace_allreduce {
    int         subgraph;
    int         n_devices;
    std::string method;
    std::string name;
    std::string type;
    int64_t     ne[4];
    size_t      nbytes;
    int64_t     t_begin;
    int64_t     t_end;

    std::vector<trace_transfer> transfers;

    // counted even when the event cap stops rows from being stored
    size_t n_transfers    = 0;
    size_t transfer_bytes = 0;
    size_t n_rpc          = 0;
    size_t rpc_bytes      = 0;
};

struct endpoint_totals {
    size_t n_calls   = 0;
    size_t bytes_out = 0;
    size_t bytes_in  = 0;
};

struct trace_step {
    std::string label;
    int         n_tokens;
    int64_t     t_begin;
    int64_t     t_end;

    std::vector<trace_allreduce> allreduces;
    std::vector<trace_rpc>       rpcs;

    size_t n_rpc          = 0;
    size_t n_rpc_dropped  = 0;
    size_t bytes_out      = 0;
    size_t bytes_in       = 0;

    size_t n_transfers         = 0;
    size_t n_transfers_dropped = 0;
    size_t transfer_bytes      = 0;

    std::map<std::string, endpoint_totals> by_cmd;
};

struct trace_state {
    std::mutex mutex;

    bool   enabled    = false;
    bool   written    = false;
    size_t max_steps  = 64;
    size_t max_events = 4096; // rows stored per bucket; totals stay exact past this

    std::string path;

    std::vector<trace_step> steps;
    trace_step              preamble; // events recorded outside any decode step
    bool                    in_step = false;
    bool                    step_detail = false; // false once the detail cap is hit

    // aggregates, kept even after the detail cap is hit
    size_t n_steps      = 0;
    size_t n_allreduce  = 0;
    size_t n_butterfly  = 0;
    size_t n_native     = 0;
    size_t n_transfers  = 0;
    size_t n_rpc        = 0;
    size_t bytes_out    = 0;
    size_t bytes_in     = 0;
    size_t transfer_bytes = 0;

    std::map<std::string, endpoint_totals> endpoints;

    int64_t t_start = 0;
};

trace_state    g_trace;
std::once_flag g_trace_once;

void trace_init() {
    const char * env = getenv("GGML_TRACE");
    g_trace.enabled = env != nullptr && atoi(env) != 0;
    if (!g_trace.enabled) {
        return;
    }

    const char * path = getenv("GGML_TRACE_HTML");
    g_trace.path = path ? path : "llama-trace.html";

    const char * max_steps = getenv("GGML_TRACE_MAX_STEPS");
    if (max_steps) {
        const int v = atoi(max_steps);
        g_trace.max_steps = v > 0 ? (size_t) v : 0;
    }

    const char * max_events = getenv("GGML_TRACE_MAX_EVENTS");
    if (max_events) {
        const int v = atoi(max_events);
        g_trace.max_events = v > 0 ? (size_t) v : 0;
    }

    g_trace.preamble.label    = "outside decode (load / warmup)";
    g_trace.preamble.n_tokens = 0;
    g_trace.preamble.t_begin  = ggml_time_us();
    g_trace.preamble.t_end    = g_trace.preamble.t_begin;

    g_trace.t_start = ggml_time_us();

    atexit(ggml_trace_write_html_default);

    fprintf(stderr, "ggml_trace: enabled, report will be written to %s\n", g_trace.path.c_str());
}

bool trace_active() {
    std::call_once(g_trace_once, trace_init);
    return g_trace.enabled;
}

// caller holds the lock; returns the bucket that new events belong to,
// or nullptr when the detail cap has been reached
trace_step * cur_step() {
    if (!g_trace.in_step) {
        g_trace.preamble.t_end = ggml_time_us();
        return &g_trace.preamble;
    }
    if (!g_trace.step_detail || g_trace.steps.empty()) {
        return nullptr;
    }
    return &g_trace.steps.back();
}

void copy_ne(int64_t dst[4], const int64_t * src) {
    for (int i = 0; i < 4; i++) {
        dst[i] = src ? src[i] : 0;
    }
}

std::string fmt_bytes(size_t n) {
    char buf[64];
    if (n >= 1024ull*1024ull*1024ull) {
        snprintf(buf, sizeof(buf), "%.2f GiB", double(n)/(1024.0*1024.0*1024.0));
    } else if (n >= 1024ull*1024ull) {
        snprintf(buf, sizeof(buf), "%.2f MiB", double(n)/(1024.0*1024.0));
    } else if (n >= 1024ull) {
        snprintf(buf, sizeof(buf), "%.2f KiB", double(n)/1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", n);
    }
    return buf;
}

std::string fmt_ne(const int64_t ne[4]) {
    char buf[128];
    int  n = 4;
    while (n > 1 && ne[n-1] <= 1) {
        n--;
    }
    std::string s = "[";
    for (int i = 0; i < n; i++) {
        snprintf(buf, sizeof(buf), "%s%" PRId64, i ? ", " : "", ne[i]);
        s += buf;
    }
    return s + "]";
}

std::string esc(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        switch (c) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            default:   out += c;        break;
        }
    }
    return out;
}

const char * css = R"(
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: #f6f7f9; color: #1c1e21; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
.sub { color: #666; font-size: 13px; margin-bottom: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
.card { background: #fff; border: 1px solid #dcdfe4; border-radius: 8px; padding: 12px 16px; min-width: 150px; }
.card .k { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }
.card .v { font-size: 20px; font-weight: 600; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; background: #fff; }
th, td { border: 1px solid #e3e6ea; padding: 4px 8px; text-align: left; }
th { background: #eef1f4; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
details { background: #fff; border: 1px solid #dcdfe4; border-radius: 6px; margin-bottom: 6px; }
details[open] { padding-bottom: 8px; }
summary { cursor: pointer; padding: 8px 12px; font-size: 13px; user-select: none; }
summary:hover { background: #f0f2f5; }
details details { margin: 6px 12px; background: #fbfcfd; }
.inner { padding: 0 12px; }
.tag { display: inline-block; border-radius: 4px; padding: 1px 6px; font-size: 11px; font-weight: 600; margin-left: 6px; }
.tag.butterfly { background: #fde8e8; color: #9b1c1c; }
.tag.native { background: #e6f4ea; color: #1e6b34; }
.bar { display: inline-block; height: 8px; background: #4a7dbd; border-radius: 2px; vertical-align: middle; }
.muted { color: #6b7280; }
button { font-size: 12px; padding: 5px 10px; margin-right: 6px; border: 1px solid #c3c8ce; background: #fff; border-radius: 5px; cursor: pointer; }
button:hover { background: #eef1f4; }
h2 { font-size: 15px; margin: 24px 0 8px 0; }
h3 { font-size: 12px; margin: 8px 0 4px 0; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
)";

void write_transfers(FILE * f, const trace_allreduce & ar) {
    if (ar.transfers.empty()) {
        return;
    }
    fprintf(f, "<div class=\"inner\"><table><tr><th>#</th><th>phase</th><th>from</th><th>to</th><th>tensor</th><th>shape</th><th>bytes</th></tr>\n");
    for (size_t i = 0; i < ar.transfers.size(); i++) {
        const trace_transfer & t = ar.transfers[i];
        fprintf(f, "<tr><td class=\"num\">%zu</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class=\"num\">%s</td></tr>\n",
            i, esc(t.phase).c_str(), esc(t.src_name).c_str(), esc(t.dst_name).c_str(),
            esc(t.name).c_str(), fmt_ne(t.ne).c_str(), fmt_bytes(t.nbytes).c_str());
    }
    fprintf(f, "</table>\n");
    if (ar.n_transfers > ar.transfers.size()) {
        fprintf(f, "<p class=\"muted\">%zu further transfers are included in the totals but not listed.</p>\n",
            ar.n_transfers - ar.transfers.size());
    }
    fprintf(f, "</div>\n");
}

void write_rpcs(FILE * f, const trace_step & s, int allreduce) {
    size_t n = 0;
    for (const trace_rpc & r : s.rpcs) {
        n += r.allreduce == allreduce ? 1 : 0;
    }
    if (n == 0) {
        return;
    }
    fprintf(f, "<div class=\"inner\"><table><tr><th>#</th><th>endpoint</th><th>dev</th><th>cmd</th><th>tensor</th><th>shape</th><th>out</th><th>in</th></tr>\n");
    size_t i = 0;
    for (const trace_rpc & r : s.rpcs) {
        if (r.allreduce != allreduce) {
            continue;
        }
        fprintf(f, "<tr><td class=\"num\">%zu</td><td>%s</td><td class=\"num\">%d</td><td>%s</td><td>%s</td><td>%s</td><td class=\"num\">%s</td><td class=\"num\">%s</td></tr>\n",
            i++, esc(r.endpoint).c_str(), r.device, esc(r.cmd).c_str(), esc(r.name).c_str(),
            fmt_ne(r.ne).c_str(), fmt_bytes(r.bytes_out).c_str(), fmt_bytes(r.bytes_in).c_str());
    }
    fprintf(f, "</table>\n");
    if (allreduce == -1 && s.n_rpc_dropped) {
        fprintf(f, "<p class=\"muted\">%zu further RPC calls are included in the totals but not listed; raise GGML_TRACE_MAX_EVENTS to list them.</p>\n", s.n_rpc_dropped);
    }
    fprintf(f, "</div>\n");
}

void write_by_cmd(FILE * f, const trace_step & s) {
    if (s.by_cmd.size() < 2) {
        return;
    }
    fprintf(f, "<div class=\"inner\"><h3>by command</h3><table><tr><th>cmd</th><th>calls</th><th>sent</th><th>received</th></tr>\n");
    for (const auto & kv : s.by_cmd) {
        fprintf(f, "<tr><td>%s</td><td class=\"num\">%zu</td><td class=\"num\">%s</td><td class=\"num\">%s</td></tr>\n",
            esc(kv.first).c_str(), kv.second.n_calls,
            fmt_bytes(kv.second.bytes_out).c_str(), fmt_bytes(kv.second.bytes_in).c_str());
    }
    fprintf(f, "</table></div>\n");
}

void write_step(FILE * f, const trace_step & s, size_t max_step_bytes) {
    const size_t bytes = s.bytes_out + s.bytes_in;

    size_t n_butterfly = 0;
    for (const trace_allreduce & a : s.allreduces) {
        n_butterfly += a.method == "butterfly" ? 1 : 0;
    }

    const int width = max_step_bytes ? int(200.0*double(bytes)/double(max_step_bytes)) : 0;

    fprintf(f, "<details><summary><b>%s</b> <span class=\"muted\">- %zu allreduce (%zu butterfly), %zu transfers (%s), %zu RPC calls (%s on the wire), %.2f ms</span> <span class=\"bar\" style=\"width:%dpx\"></span></summary>\n",
        esc(s.label).c_str(), s.allreduces.size(), n_butterfly, s.n_transfers, fmt_bytes(s.transfer_bytes).c_str(),
        s.n_rpc, fmt_bytes(bytes).c_str(), double(s.t_end - s.t_begin)/1000.0,
        width < 1 && bytes ? 1 : width);

    write_by_cmd(f, s);

    for (size_t i = 0; i < s.allreduces.size(); i++) {
        const trace_allreduce & a = s.allreduces[i];
        fprintf(f, "<details><summary>allreduce #%zu <span class=\"tag %s\">%s</span> <span class=\"muted\">subgraph %d, %d devices, %s %s %s, %zu transfers (%s), %zu RPC (%s), %.3f ms</span></summary>\n",
            i, esc(a.method).c_str(), esc(a.method).c_str(), a.subgraph, a.n_devices,
            esc(a.name).c_str(), fmt_ne(a.ne).c_str(), fmt_bytes(a.nbytes).c_str(),
            a.n_transfers, fmt_bytes(a.transfer_bytes).c_str(),
            a.n_rpc, fmt_bytes(a.rpc_bytes).c_str(),
            double(a.t_end - a.t_begin)/1000.0);
        write_transfers(f, a);
        write_rpcs(f, s, (int) i);
        fprintf(f, "</details>\n");
    }

    write_rpcs(f, s, -1);
    fprintf(f, "</details>\n");
}

} // namespace

bool ggml_trace_enabled(void) {
    return trace_active();
}

void ggml_trace_step_begin(const char * label, int n_tokens) {
    if (!trace_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    g_trace.n_steps++;
    g_trace.in_step = true;

    if (g_trace.steps.size() >= g_trace.max_steps) {
        g_trace.step_detail = false;
        return;
    }
    g_trace.step_detail = true;

    trace_step s;
    char buf[128];
    snprintf(buf, sizeof(buf), "step %zu - %s (%d token%s)",
        g_trace.n_steps - 1, label ? label : "decode", n_tokens, n_tokens == 1 ? "" : "s");
    s.label    = buf;
    s.n_tokens = n_tokens;
    s.t_begin  = ggml_time_us();
    s.t_end    = s.t_begin;
    g_trace.steps.push_back(std::move(s));
}

void ggml_trace_step_end(void) {
    if (!trace_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    trace_step * s = cur_step();
    if (s) {
        s->t_end = ggml_time_us();
    }
    g_trace.in_step = false;
}

int ggml_trace_allreduce_begin(int subgraph, int n_devices, const char * name,
                               const char * type, const int64_t * ne, size_t nbytes) {
    if (!trace_active()) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    g_trace.n_allreduce++;

    trace_step * s = cur_step();
    if (!s) {
        return -1;
    }

    trace_allreduce a;
    a.subgraph  = subgraph;
    a.n_devices = n_devices;
    a.method    = "?";
    a.name      = name ? name : "";
    a.type      = type ? type : "";
    a.nbytes    = nbytes;
    a.t_begin   = ggml_time_us();
    a.t_end     = a.t_begin;
    copy_ne(a.ne, ne);

    s->allreduces.push_back(std::move(a));
    return (int) s->allreduces.size() - 1;
}

void ggml_trace_allreduce_end(int id, const char * method) {
    if (!trace_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    const bool butterfly = method && strcmp(method, "butterfly") == 0;
    if (butterfly) {
        g_trace.n_butterfly++;
    } else {
        g_trace.n_native++;
    }

    trace_step * s = cur_step();
    if (!s || id < 0 || id >= (int) s->allreduces.size()) {
        return;
    }
    s->allreduces[id].method = method ? method : "?";
    s->allreduces[id].t_end  = ggml_time_us();
}

void ggml_trace_transfer(const char * phase, const char * src_name, const char * dst_name,
                         const char * name, const int64_t * ne, size_t nbytes) {
    if (!trace_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    g_trace.n_transfers++;
    g_trace.transfer_bytes += nbytes;

    trace_step * s = cur_step();
    if (!s || s->allreduces.empty()) {
        return;
    }

    trace_allreduce & a = s->allreduces.back();

    s->n_transfers++;
    s->transfer_bytes += nbytes;
    a.n_transfers++;
    a.transfer_bytes += nbytes;

    if (a.transfers.size() >= g_trace.max_events) {
        s->n_transfers_dropped++;
        return;
    }

    trace_transfer t;
    t.phase    = phase ? phase : "";
    t.src_name = src_name ? src_name : "";
    t.dst_name = dst_name ? dst_name : "";
    t.name     = name ? name : "";
    t.nbytes   = nbytes;
    copy_ne(t.ne, ne);

    a.transfers.push_back(std::move(t));
}

void ggml_trace_rpc(const char * endpoint, int device, const char * cmd,
                    const char * name, const int64_t * ne,
                    size_t bytes_out, size_t bytes_in) {
    if (!trace_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    g_trace.n_rpc++;
    g_trace.bytes_out += bytes_out;
    g_trace.bytes_in  += bytes_in;

    endpoint_totals & tot = g_trace.endpoints[endpoint ? endpoint : "?"];
    tot.n_calls++;
    tot.bytes_out += bytes_out;
    tot.bytes_in  += bytes_in;

    trace_step * s = cur_step();
    if (!s) {
        return;
    }

    int i_ar = s->allreduces.empty() ? -1 : (int) s->allreduces.size() - 1;

    // an allreduce that already ended does not own subsequent traffic
    if (i_ar >= 0 && s->allreduces[i_ar].method != "?") {
        i_ar = -1;
    }

    s->n_rpc++;
    s->bytes_out += bytes_out;
    s->bytes_in  += bytes_in;

    endpoint_totals & ct = s->by_cmd[cmd ? cmd : "?"];
    ct.n_calls++;
    ct.bytes_out += bytes_out;
    ct.bytes_in  += bytes_in;

    if (i_ar >= 0) {
        s->allreduces[i_ar].n_rpc++;
        s->allreduces[i_ar].rpc_bytes += bytes_out + bytes_in;
    }

    if (s->rpcs.size() >= g_trace.max_events) {
        s->n_rpc_dropped++;
        return;
    }

    trace_rpc r;
    r.endpoint  = endpoint ? endpoint : "?";
    r.device    = device;
    r.cmd       = cmd ? cmd : "";
    r.name      = name ? name : "";
    r.bytes_out = bytes_out;
    r.bytes_in  = bytes_in;
    r.allreduce = i_ar;
    copy_ne(r.ne, ne);

    s->rpcs.push_back(std::move(r));
}

void ggml_trace_write_html(const char * path) {
    if (!trace_active() || !path) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_trace.mutex);

    // the CLI writes on /exit, the atexit handler is only a fallback
    if (g_trace.written) {
        return;
    }
    g_trace.written = true;

    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "ggml_trace: failed to open %s for writing\n", path);
        return;
    }

    const double elapsed_s = double(ggml_time_us() - g_trace.t_start)/1e6;

    size_t max_step_bytes = 0;
    size_t decode_bytes   = 0;
    for (const trace_step & s : g_trace.steps) {
        const size_t b = s.bytes_out + s.bytes_in;
        max_step_bytes = b > max_step_bytes ? b : max_step_bytes;
        decode_bytes  += b;
    }

    fprintf(f, "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n");
    fprintf(f, "<title>llama.cpp tensor-parallel / RPC trace</title>\n<style>%s</style></head><body>\n", css);
    fprintf(f, "<h1>llama.cpp tensor-parallel / RPC trace</h1>\n");
    fprintf(f, "<div class=\"sub\">%.2f s wall clock, %zu decode steps recorded (detail cap %zu)</div>\n",
        elapsed_s, g_trace.n_steps, g_trace.max_steps);

    fprintf(f, "<div class=\"cards\">\n");
    fprintf(f, "<div class=\"card\"><div class=\"k\">decode steps</div><div class=\"v\">%zu</div></div>\n", g_trace.n_steps);
    fprintf(f, "<div class=\"card\"><div class=\"k\">allreduces</div><div class=\"v\">%zu</div></div>\n", g_trace.n_allreduce);
    fprintf(f, "<div class=\"card\"><div class=\"k\">butterfly fallback</div><div class=\"v\">%zu</div></div>\n", g_trace.n_butterfly);
    fprintf(f, "<div class=\"card\"><div class=\"k\">native comm</div><div class=\"v\">%zu</div></div>\n", g_trace.n_native);
    fprintf(f, "<div class=\"card\"><div class=\"k\">butterfly transfers</div><div class=\"v\">%zu</div></div>\n", g_trace.n_transfers);
    fprintf(f, "<div class=\"card\"><div class=\"k\">transfer volume</div><div class=\"v\">%s</div></div>\n", fmt_bytes(g_trace.transfer_bytes).c_str());
    fprintf(f, "<div class=\"card\"><div class=\"k\">RPC calls</div><div class=\"v\">%zu</div></div>\n", g_trace.n_rpc);
    fprintf(f, "<div class=\"card\"><div class=\"k\">sent to workers</div><div class=\"v\">%s</div></div>\n", fmt_bytes(g_trace.bytes_out).c_str());
    fprintf(f, "<div class=\"card\"><div class=\"k\">received from workers</div><div class=\"v\">%s</div></div>\n", fmt_bytes(g_trace.bytes_in).c_str());
    if (g_trace.n_steps > 0) {
        fprintf(f, "<div class=\"card\"><div class=\"k\">wire bytes / step</div><div class=\"v\">%s</div></div>\n",
            fmt_bytes(decode_bytes / (g_trace.steps.empty() ? 1 : g_trace.steps.size())).c_str());
        fprintf(f, "<div class=\"card\"><div class=\"k\">allreduce / step</div><div class=\"v\">%.1f</div></div>\n",
            double(g_trace.n_allreduce)/double(g_trace.n_steps));
    }
    fprintf(f, "</div>\n");

    fprintf(f, "<h2>Per endpoint</h2>\n<table><tr><th>endpoint</th><th>calls</th><th>sent</th><th>received</th><th>total</th></tr>\n");
    for (const auto & kv : g_trace.endpoints) {
        fprintf(f, "<tr><td>%s</td><td class=\"num\">%zu</td><td class=\"num\">%s</td><td class=\"num\">%s</td><td class=\"num\">%s</td></tr>\n",
            esc(kv.first).c_str(), kv.second.n_calls,
            fmt_bytes(kv.second.bytes_out).c_str(), fmt_bytes(kv.second.bytes_in).c_str(),
            fmt_bytes(kv.second.bytes_out + kv.second.bytes_in).c_str());
    }
    fprintf(f, "</table>\n");

    fprintf(f, "<h2>Timeline</h2>\n");
    fprintf(f, "<p><button onclick=\"document.querySelectorAll('details').forEach(d=>d.open=true)\">expand all</button>");
    fprintf(f, "<button onclick=\"document.querySelectorAll('details').forEach(d=>d.open=false)\">collapse all</button></p>\n");

    if (!g_trace.preamble.rpcs.empty() || g_trace.preamble.n_rpc > 0) {
        write_step(f, g_trace.preamble, max_step_bytes);
    }
    for (size_t i = 0; i < g_trace.steps.size(); i++) {
        write_step(f, g_trace.steps[i], max_step_bytes);
    }

    if (g_trace.n_steps > g_trace.steps.size()) {
        fprintf(f, "<p class=\"muted\">%zu further decode steps were counted in the totals above but not recorded in detail; raise GGML_TRACE_MAX_STEPS to capture them.</p>\n",
            g_trace.n_steps - g_trace.steps.size());
    }

    fprintf(f, "</body></html>\n");
    fclose(f);

    fprintf(stderr, "ggml_trace: wrote %s (%zu decode steps, %zu allreduce, %zu butterfly, %zu RPC calls)\n",
        path, g_trace.n_steps, g_trace.n_allreduce, g_trace.n_butterfly, g_trace.n_rpc);
}

void ggml_trace_write_html_default(void) {
    if (!trace_active()) {
        return;
    }
    ggml_trace_write_html(g_trace.path.c_str());
}
