#include "server-models-sched.h"
#include "server-models.h"

#include "fit.h"

#include <algorithm>
#include <cinttypes>

static constexpr int MEASURE_TIMEOUT = 300; // seconds, then the measure child is killed

//
// server_model_memory
//

server_model_memory::server_model_memory(uint32_t n_ctx, const std::vector<common_memory_breakdown_row> & rows) : n_ctx(n_ctx) {
    for (const auto & row : rows) {
        device_t d;
        switch (row.kind) {
            case common_memory_breakdown_row::DEVICE:      d.kind = "device";      break;
            case common_memory_breakdown_row::HOST:        d.kind = "host";        break;
            case common_memory_breakdown_row::BUFFER_TYPE: d.kind = "buffer_type"; break;
        }
        d.name    = row.name;
        d.total   = row.total;
        d.free    = row.free;
        d.model   = row.model;
        d.context = row.context;
        d.compute = row.compute;
        devices.push_back(std::move(d));
    }
}

json server_model_memory::to_json() const {
    json devs = json::array();
    for (const auto & d : devices) {
        json jd = {
            {"kind",    d.kind},
            {"name",    d.name},
            {"model",   d.model},
            {"context", d.context},
            {"compute", d.compute},
        };
        if (d.is_device()) {
            jd["total"] = d.total;
            jd["free"]  = d.free;
        }
        devs.push_back(std::move(jd));
    }
    return {
        {"n_ctx",   n_ctx},
        {"devices", devs},
    };
}

server_model_memory server_model_memory::from_json(const json & j) {
    server_model_memory mem;
    mem.n_ctx = j.at("n_ctx").get<uint32_t>();
    for (const auto & jd : j.at("devices")) {
        device_t d;
        d.kind    = jd.at("kind").get<std::string>();
        d.name    = jd.at("name").get<std::string>();
        d.total   = json_value(jd, "total", (size_t) 0);
        d.free    = json_value(jd, "free",  (size_t) 0);
        d.model   = jd.at("model").get<size_t>();
        d.context = jd.at("context").get<size_t>();
        d.compute = jd.at("compute").get<size_t>();
        mem.devices.push_back(std::move(d));
    }
    return mem;
}

json server_model_memory_info::to_json() const {
    return {
        {"status",    server_model_memory_status_to_string(status)},
        {"min",       min       ? min->to_json()       : json(nullptr)},
        {"max",       max       ? max->to_json()       : json(nullptr)},
        {"effective", effective ? effective->to_json() : json(nullptr)},
    };
}

//
// request queue
//

bool server_models_sched::has_capacity(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    return models.base_params.models_max <= 0
        || count_running() < (size_t) models.base_params.models_max;
}

std::string server_models_sched::pick_victim(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    std::string victim;
    int64_t victim_last_used = 0;
    for (const auto & m : models.mapping) {
        // a busy model is mid-request, one still coming up has no request to finish
        if (m.second.req_count != 0 || !m.second.meta.is_ready_or_sleep()) {
            continue;
        }
        // already on its way out, or a queued request wants it
        if (models.stopping_models.count(m.first) || find(m.first)) {
            continue;
        }
        if (victim.empty() || m.second.meta.last_used < victim_last_used) {
            victim           = m.first;
            victim_last_used = m.second.meta.last_used;
        }
    }
    return victim;
}

void server_models_sched::join(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
    check_lock(lk);
    if (entry_t * e = find(model_id)) {
        e->n_waiters++;
        SRV_INF("request for name=%s joined the queue, %d waiting\n", model_id.c_str(), e->n_waiters);
        return;
    }
    queue.push_back({ model_id, 1, false });
    SRV_INF("request for name=%s queued at position %zu\n",
            model_id.c_str(), queue.size());
}

void server_models_sched::leave(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
    check_lock(lk);
    for (auto it = queue.begin(); it != queue.end(); ++it) {
        if (it->model_id == model_id) {
            if (--it->n_waiters <= 0) {
                queue.erase(it); // last one waiting for this model went away
            }
            return;
        }
    }
}

bool server_models_sched::queue_empty(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    return queue.empty();
}

bool server_models_sched::try_claim(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
    check_lock(lk);
    if (queue.empty() || queue.front().model_id != model_id || queue.front().loading) {
        return false;
    }
    if (!has_capacity(lk)) {
        return false;
    }
    // over the memory margin with every running model busy: wait for one to go idle
    if (memory_enabled() && limits_exceeded(lk, model_id, /* verbose */ false) && pick_victim(lk).empty()) {
        return false;
    }
    queue.front().loading = true;
    return true;
}

void server_models_sched::claim_done(std::unique_lock<std::mutex> & lk, const std::string & model_id, bool ok) {
    check_lock(lk);
    if (ok) {
        return;
    }
    for (auto it = queue.begin(); it != queue.end(); ++it) {
        if (it->model_id == model_id) {
            it->loading = false;
            return;
        }
    }
}

void server_models_sched::tick(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    if (models.base_params.models_max <= 0 || queue.empty()) {
        return;
    }
    int n_running  = 0;
    int n_stopping = 0;
    for (const auto & m : models.mapping) {
        if (m.second.meta.is_running()) {
            n_running++;
            if (models.stopping_models.count(m.first)) {
                n_stopping++;
            }
        }
    }
    int n_needed  = 0;
    int n_claimed = 0; // claimed the slot, but load() has not spawned yet
    for (const auto & e : queue) {
        if (!e.loading) {
            n_needed++;
            continue;
        }
        auto it = models.mapping.find(e.model_id);
        if (it != models.mapping.end() && !it->second.meta.is_running()) {
            n_claimed++;
        }
    }
    int n_free = models.base_params.models_max - n_running + n_stopping - n_claimed;
    while (n_free < n_needed) {
        std::string victim = pick_victim(lk);
        if (victim.empty()) {
            return; // all remaining models are busy, wait for a request to end
        }
        SRV_INF("evicting idle LRU name=%s for a queued request\n", victim.c_str());
        models.request_stop(victim);
        n_free++;
    }
}

server_models_sched::entry_t * server_models_sched::find(const std::string & model_id) {
    for (auto & e : queue) {
        if (e.model_id == model_id) {
            return &e;
        }
    }
    return nullptr;
}

void server_models_sched::check_lock(std::unique_lock<std::mutex> & lk) {
    GGML_ASSERT(lk.owns_lock() && lk.mutex() == &models.mutex);
}

size_t server_models_sched::count_running() {
    size_t count = 0;
    for (const auto & m : models.mapping) {
        if (m.second.meta.is_running()) {
            count++;
        }
    }
    return count;
}

//
// loading a model
//

bool server_models_sched::limits_exceeded(std::unique_lock<std::mutex> & lk, const std::string & name, bool verbose) {
    if (!has_capacity(lk)) {
        return true;
    }
    if (!memory_enabled()) {
        return false;
    }
    const auto missing = deficit(name);
    if (missing.empty()) {
        return false;
    }
    bool any_running = false;
    for (const auto & m : models.mapping) {
        if (m.second.meta.is_running()) {
            any_running = true;
            break;
        }
    }
    for (const auto & [dev, bytes] : missing) {
        if (!verbose) {
            break;
        }
        if (any_running) {
            SRV_DBG("model name=%s is short by %zu MiB on %s\n", name.c_str(), bytes / (1024 * 1024), dev.c_str());
        } else {
            // nothing to unload: min is pessimistic, let --fit in the instance try
            SRV_WRN("model name=%s is projected to be short by %zu MiB on %s with nothing else loaded, loading anyway\n",
                name.c_str(), bytes / (1024 * 1024), dev.c_str());
        }
    }
    return any_running;
}

void server_models_sched::make_room(const std::string & name) {
    if (memory_enabled()) {
        // the measure result decides what to unload
        std::unique_lock<std::mutex> lk(models.mutex);
        enqueue_measure(name, /* front */ true);
        auto measured = [this, &name]() {
            auto it = models.mapping.find(name);
            return it == models.mapping.end() || it->second.meta.memory.is_settled();
        };
        if (!measured()) {
            SRV_INF("waiting for the memory measurement of model name=%s\n", name.c_str());
            models.cv.wait(lk, measured);
        }
    }

    while (true) {
        std::string lru_model_name;
        {
            std::unique_lock<std::mutex> lk(models.mutex);
            if (!limits_exceeded(lk, name)) {
                return;
            }
            lru_model_name = pick_victim(lk);
        }
        if (lru_model_name.empty()) {
            return; // every running model is busy, load() reports the limit
        }
        SRV_INF("limits reached, removing LRU name=%s\n", lru_model_name.c_str());
        models.unload(lru_model_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(models.mutex);
            models.cv.wait(lk, [this, &lru_model_name]() {
                return models.mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}

//
// memory accounting
//

bool server_models_sched::memory_enabled() const {
    return models.base_params.models_memory_margin > 0;
}

void server_models_sched::measure_unmeasured(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    for (const auto & [name, inst] : models.mapping) {
        enqueue_measure(name);
    }
}

void server_models_sched::on_spawned(std::unique_lock<std::mutex> & lk, const std::string & name) {
    check_lock(lk);
    reserve(name);
}

void server_models_sched::on_ready(std::unique_lock<std::mutex> & lk, const std::string & name, const std::optional<server_model_memory> & memory) {
    check_lock(lk);
    if (!memory_enabled()) {
        return;
    }
    auto it = models.mapping.find(name);
    if (it == models.mapping.end()) {
        return;
    }
    if (!memory.has_value()) {
        reserve(name); // woke up from sleep
        return;
    }
    // use the real usage, not the estimate
    auto & info = it->second.meta.memory;
    info.effective = memory;
    release(name);
    reserve(name);
    calibrate(*memory, name);
    if (info.status == SERVER_MODEL_MEMORY_FAILED && info.n_attempts < 2) {
        // the model was maybe not downloaded at measure time, retry
        info.status = SERVER_MODEL_MEMORY_UNMEASURED;
        enqueue_measure(name);
    }
}

void server_models_sched::on_sleeping(std::unique_lock<std::mutex> & lk, const std::string & name) {
    check_lock(lk);
    release(name);
}

void server_models_sched::on_unloaded(std::unique_lock<std::mutex> & lk, const std::string & name) {
    check_lock(lk);
    release(name);
    auto it = models.mapping.find(name);
    if (it != models.mapping.end()) {
        it->second.meta.memory.effective.reset();
    }
}

void server_models_sched::cancel_measurements(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    measure_queue.clear();
    if (measure_proc) {
        SRV_INF("stopping measure process for model name=%s\n", measuring.c_str());
        measure_proc->terminate();
        // the monitor reaps it and clears measuring
    }
}

bool server_models_sched::is_measuring(std::unique_lock<std::mutex> & lk) {
    check_lock(lk);
    return !measuring.empty();
}

void server_models_sched::enqueue_measure(const std::string & name, bool front) {
    if (!memory_enabled()) {
        return;
    }
    auto it = models.mapping.find(name);
    if (it == models.mapping.end() || it->second.meta.memory.status != SERVER_MODEL_MEMORY_UNMEASURED) {
        return;
    }
    auto pos = std::find(measure_queue.begin(), measure_queue.end(), name);
    if (pos != measure_queue.end()) {
        if (!front) {
            return; // already queued
        }
        measure_queue.erase(pos);
    }
    if (front) {
        measure_queue.push_front(name);
    } else {
        measure_queue.push_back(name);
    }
    measure_next();
}

void server_models_sched::measure_next() {
    if (!measuring.empty()) {
        return; // one measure child at a time
    }
    while (!measure_queue.empty()) {
        std::string name = std::move(measure_queue.front());
        measure_queue.pop_front();
        auto it = models.mapping.find(name);
        if (it == models.mapping.end() || it->second.meta.memory.status != SERVER_MODEL_MEMORY_UNMEASURED) {
            continue; // removed or measured meanwhile
        }
        auto & meta = it->second.meta;

        // same args as the instance, --offline: never download
        std::vector<std::string> child_args = meta.args;
        child_args.push_back("--offline");
        std::vector<std::string> child_env = models.base_env;
        child_env.push_back(std::string(SERVER_ENV_ROUTER_PORT) + "=" + std::to_string(models.base_params.port));
        child_env.push_back(std::string(SERVER_ENV_CHILD_ROLE) + "=" + server_role_type_to_str(SERVER_ROLE_MEASURE));

        auto proc = std::make_shared<server_subproc>();
        if (!proc->sproc.create(child_args, subprocess_option_no_window, child_env)) {
            SRV_ERR("failed to spawn measure process for model name=%s\n", name.c_str());
            meta.memory.status = SERVER_MODEL_MEMORY_FAILED;
            continue;
        }
        SRV_INF("measuring memory for model name=%s\n", name.c_str());
        meta.memory.status = SERVER_MODEL_MEMORY_MEASURING;
        meta.memory.n_attempts++;
        measuring    = name;
        measure_proc = proc;
        models.watch_child(name, proc, SERVER_ROLE_MEASURE, 0, MEASURE_TIMEOUT);
        return;
    }
}

void server_models_sched::on_measure_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, int exit_code) {
    std::lock_guard<std::mutex> lk(models.mutex);
    if (measure_proc != proc) {
        return; // not the child we are waiting for
    }
    measure_proc.reset();
    measuring.clear();
    auto it = models.mapping.find(name);
    if (it != models.mapping.end() && it->second.meta.memory.status == SERVER_MODEL_MEMORY_MEASURING) {
        // exited without a result: crashed, or killed after the timeout
        SRV_WRN("measure process for model name=%s exited with status %d before reporting, memory limits will not apply to it\n",
            name.c_str(), exit_code);
        it->second.meta.memory.status = SERVER_MODEL_MEMORY_FAILED;
        models.notify_sse("model_memory", name, it->second.meta.memory.to_json());
    }
    models.cv.notify_all();
    measure_next();
}

void server_models_sched::handle_measured(const std::string & name, const json & payload) {
    std::lock_guard<std::mutex> lk(models.mutex);
    auto it = models.mapping.find(name);
    if (it == models.mapping.end()) {
        return;
    }
    auto & info = it->second.meta.memory;
    const std::string result = json_value(payload, "result", std::string());

    bool ok = result == "measured";
    if (ok) {
        try {
            info.min = server_model_memory::from_json(payload.at("min"));
            info.max = server_model_memory::from_json(payload.at("max"));
        } catch (const std::exception & e) {
            SRV_ERR("malformed memory measurement for model name=%s: %s\n", name.c_str(), e.what());
            info.min.reset();
            info.max.reset();
            ok = false;
        }
    } else {
        SRV_WRN("measuring memory for model name=%s failed: %s\n",
            name.c_str(), json_value(payload, "error", std::string("unknown error")).c_str());
    }

    if (ok) {
        info.status = SERVER_MODEL_MEMORY_MEASURED;
        calibrate(*info.max, ""); // latest view of the devices
        const auto & mn = *info.min;
        const auto & mx = *info.max;
        for (size_t i = 0; i < mn.devices.size() && i < mx.devices.size(); i++) {
            if (!mn.devices[i].is_device() || mn.devices[i].name != mx.devices[i].name) {
                continue;
            }
            SRV_INF("model name=%s needs on %s: %zu MiB at n_ctx=%u (min), %zu MiB at n_ctx=%u (max), %zu MiB free\n",
                name.c_str(), mn.devices[i].name.c_str(),
                mn.devices[i].self() / (1024 * 1024), mn.n_ctx,
                mx.devices[i].self() / (1024 * 1024), mx.n_ctx,
                mx.devices[i].free   / (1024 * 1024));
        }
    } else {
        info.status = SERVER_MODEL_MEMORY_FAILED;
    }
    models.notify_sse("model_memory", name, info.to_json());
    models.cv.notify_all();
}

void server_models_sched::calibrate(const server_model_memory & mem, const std::string & reporter) {
    for (const auto & d : mem.devices) {
        if (!d.is_device() || d.total == 0) {
            continue;
        }
        // other instances are tracked, so they are not external (CUDA counts them in free, Metal does not)
        int64_t tracked = 0;
        for (const auto & [name, devs] : held) {
            if (name == reporter) {
                continue;
            }
            auto h = devs.find(d.name);
            if (h != devs.end()) {
                tracked += (int64_t) h->second;
            }
        }
        // a measure child allocates nothing, its free does not include self
        const int64_t self = reporter.empty() ? 0 : (int64_t) d.self();
        auto & dm = device_memory[d.name];
        dm.total    = (int64_t) d.total;
        dm.external = std::max<int64_t>(0, dm.total - (int64_t) d.free - self - tracked);
        SRV_DBG("device %s: total=%" PRId64 " MiB, external=%" PRId64 " MiB, available=%" PRId64 " MiB\n",
            d.name.c_str(), dm.total / (1024 * 1024), dm.external / (1024 * 1024), available(d.name) / (1024 * 1024));
    }
}

void server_models_sched::reserve(const std::string & name) {
    auto it = models.mapping.find(name);
    if (it == models.mapping.end() || held.count(name)) {
        return;
    }
    const auto & info = it->second.meta.memory;
    // prefer real usage over the estimate
    const auto & mem = info.effective ? info.effective : info.min;
    if (!mem) {
        return;
    }
    auto & devs = held[name];
    for (const auto & d : mem->devices) {
        if (d.is_device() && d.total > 0) {
            devs[d.name] = d.self();
        }
    }
}

void server_models_sched::release(const std::string & name) {
    held.erase(name);
}

int64_t server_models_sched::available(const std::string & dev) const {
    auto dm = device_memory.find(dev);
    if (dm == device_memory.end()) {
        return INT64_MAX; // never reported, nothing to check against
    }
    int64_t n_held = 0;
    for (const auto & [name, devs] : held) {
        auto h = devs.find(dev);
        if (h != devs.end()) {
            n_held += (int64_t) h->second;
        }
    }
    return dm->second.total - dm->second.external - n_held;
}

std::map<std::string, size_t> server_models_sched::deficit(const std::string & name) const {
    std::map<std::string, size_t> res;
    auto it = models.mapping.find(name);
    if (it == models.mapping.end() || !it->second.meta.memory.min) {
        return res;
    }
    const int64_t margin = (int64_t) models.base_params.models_memory_margin * 1024 * 1024;
    for (const auto & d : it->second.meta.memory.min->devices) {
        // host memory is assumed unlimited, same as --fit
        if (!d.is_device() || d.total == 0) {
            continue;
        }
        if (device_memory.find(d.name) == device_memory.end()) {
            continue;
        }
        const int64_t avail = available(d.name) - margin;
        if ((int64_t) d.self() > avail) {
            res[d.name] = (size_t) ((int64_t) d.self() - avail);
        }
    }
    return res;
}
