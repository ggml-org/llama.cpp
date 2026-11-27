#pragma once

#include "server-http.h"

// Wisp v1 relay (https://github.com/MercuryWorkshop/wisp-protocol): the TCP connections of the Linux machine that Llama Chat runs in the browser, multiplexed over one WebSocket
server_http_context::ws_handler_t wisp_make_handler(bool allow_private_urls);
