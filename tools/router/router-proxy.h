#pragma once

#include "router-config.h"

#include <cpp-httplib/httplib.h>

#include <string>

bool proxy_request(const httplib::Request & req,
                   httplib::Response &       res,
                   const std::string &       upstream_base,
                   const RouterOptions &     opts,
                   const std::vector<std::string> & proxy_endpoints,
                   const std::string &              override_path = {});
