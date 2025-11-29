#pragma once

#include <cpp-httplib/httplib.h>

#include <string>

bool proxy_request(const httplib::Request & req, httplib::Response & res, const std::string & upstream_base);
