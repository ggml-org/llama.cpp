#pragma once

#include "router-app.h"

#include <cpp-httplib/httplib.h>

void register_admin_routes(httplib::Server & server, RouterApp & app, const std::string & config_path = std::string());
