#include "../tools/cli/mcp.hpp"
#include "common.h"
#include "log.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mcp_config_file>\n", argv[0]);
        return 1;
    }

    std::string config_file = argv[1];

    printf("Testing MCP integration with config: %s\n", config_file.c_str());

    mcp_context mcp;
    std::string tool_to_run;
    json        tool_args = json::object();
    std::string servers_arg = "";

    if (argc >= 3) {
        tool_to_run = argv[2];
    }
    if (argc >= 4) {
        try {
            tool_args = json::parse(argv[3]);
        } catch (const std::exception & e) {
            fprintf(stderr, "Error parsing tool arguments JSON: %s\n", e.what());
            return 1;
        }
    }

    if (mcp.load_config(config_file, servers_arg)) {
        printf("MCP config loaded successfully.\n");

        auto tools = mcp.get_tools();
        printf("Found %zu tools.\n", tools.size());

        for (const auto & tool : tools) {
            printf("Tool: %s\n", tool.name.c_str());
            printf("  Description: %s\n", tool.description.c_str());
        }

        if (!tool_to_run.empty()) {
            printf("Calling tool %s...\n", tool_to_run.c_str());
            mcp.set_yolo(true);
            json res = mcp.call_tool(tool_to_run, tool_args);
            printf("Result: %s\n", res.dump().c_str());
        } else if (!tools.empty()) {
            printf("No tool specified. Calling first tool '%s' with empty args as smoke test...\n", tools[0].name.c_str());
             json args = json::object();
            mcp.set_yolo(true);
            json res = mcp.call_tool(tools[0].name, args);
            printf("Result: %s\n", res.dump().c_str());
        }

    } else {
        printf("Failed to load MCP config.\n");
        return 1;
    }

    // Allow some time for threads to shutdown if any
    std::this_thread::sleep_for(std::chrono::seconds(1));

    return 0;
}
