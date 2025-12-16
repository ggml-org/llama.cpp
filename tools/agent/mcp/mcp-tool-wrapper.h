#pragma once

#include "mcp-server-manager.h"

// Register all MCP tools from the server manager into the tool registry
// The manager must outlive the tool registrations (tools hold reference to manager)
void register_mcp_tools(mcp_server_manager & manager);
