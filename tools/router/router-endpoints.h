#pragma once

#include "router-app.h"

#include <cpp-httplib/httplib.h>

void register_routes(httplib::Server & server, RouterApp & app);
