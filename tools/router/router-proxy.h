#pragma once

#include "router-app.h"

#include <cpp-httplib/httplib.h>

#include <string>
#include <vector>

bool proxy_request(const httplib::Request & req,
                   httplib::Response &       res,
                   RouterApp &               app,
                   const std::string &       model_name,
                   const std::vector<std::string> & proxy_endpoints,
                   const std::string &              override_path = {});
