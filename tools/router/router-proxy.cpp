#include "router-proxy.h"

#include <cpp-httplib/httplib.h>

bool proxy_request(const httplib::Request & req, httplib::Response & res, const std::string & upstream_base) {
    if (upstream_base.empty()) {
        res.status = 502;
        res.set_content("{\"error\":\"missing upstream\"}", "application/json");
        return false;
    }

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
    return true;
}
