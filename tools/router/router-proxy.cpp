#include "router-proxy.h"

#include "log.h"

#include <cpp-httplib/httplib.h>

bool proxy_request(const httplib::Request & req, httplib::Response & res, const std::string & upstream_base) {
    if (upstream_base.empty()) {
        res.status = 502;
        res.set_content("{\"error\":\"missing upstream\"}", "application/json");
        return false;
    }

    LOG_INF("Proxying %s %s to upstream %s\n", req.method.c_str(), req.path.c_str(), upstream_base.c_str());
    httplib::Client client(upstream_base.c_str());
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(600, 0);

    httplib::Headers headers = req.headers;
    headers.erase("Host");

    const std::string path = !req.target.empty() ? req.target : req.path;

    std::string content_type = req.get_header_value("Content-Type", "application/json");

    httplib::Result result;
    if (req.method == "POST") {
        result = client.Post(path.c_str(), headers, req.body, content_type.c_str());
    } else {
        result = client.Get(path.c_str(), headers);
    }

    if (!result) {
        LOG_ERR("Upstream %s unavailable for %s %s\n", upstream_base.c_str(), req.method.c_str(), path.c_str());
        res.status = 502;
        res.set_content("{\"error\":\"upstream unavailable\"}", "application/json");
        return false;
    }

    res.status = result->status;
    res.reason = result->reason;
    for (const auto & h : result->headers) {
        res.set_header(h.first, h.second);
    }

    const auto ct = result->get_header_value("Content-Type", "application/octet-stream");
    res.set_content(result->body, ct.c_str());
    LOG_INF("Upstream response %d (%s) relayed for %s\n", res.status, res.reason.c_str(), path.c_str());
    return true;
}
