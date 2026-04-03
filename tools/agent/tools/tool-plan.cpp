#include "../tool-registry.h"

#include <sstream>
#include <string>

// ANSI color/style codes (same convention as tool-edit.cpp)
static const char * ANSI_BOLD   = "\033[1m";
static const char * ANSI_GREEN  = "\033[32m";
static const char * ANSI_YELLOW = "\033[33m";
static const char * ANSI_DIM    = "\033[2m";
static const char * ANSI_RESET  = "\033[0m";

// UTF-8 emoji sequences
static const char * ICON_COMPLETED   = "\xe2\x9c\x85"; // ✅
static const char * ICON_IN_PROGRESS = "\xf0\x9f\x94\x84"; // 🔄
static const char * ICON_PENDING     = "\xe2\xac\x9c"; // ⬜

static tool_result plan_execute(const json & args, const tool_context & /* ctx */) {
    if (!args.contains("plan") || !args["plan"].is_array()) {
        return {false, "", "plan parameter is required and must be an array"};
    }

    std::string explanation = args.value("explanation", "");

    std::ostringstream out;
    out << ANSI_BOLD << "Plan:" << ANSI_RESET << "\n";

    for (const auto & item : args["plan"]) {
        std::string step   = item.value("step", "");
        std::string status = item.value("status", "pending");

        const char * icon;
        const char * color;
        if (status == "completed") {
            icon  = ICON_COMPLETED;
            color = ANSI_GREEN;
        } else if (status == "in_progress") {
            icon  = ICON_IN_PROGRESS;
            color = ANSI_YELLOW;
        } else {
            icon  = ICON_PENDING;
            color = ANSI_DIM;
        }

        out << "  " << icon << " " << color << step << ANSI_RESET;
        if (status == "in_progress") {
            out << ANSI_DIM << "  (in progress)" << ANSI_RESET;
        }
        out << "\n";
    }

    if (!explanation.empty()) {
        out << "\n" << ANSI_DIM << explanation << ANSI_RESET << "\n";
    }

    return {true, out.str(), ""};
}

static tool_def plan_tool = {
    "update_plan",
    "Update and display the current task plan. Use for multi-step tasks to show progress "
    "while staying in the tool-calling loop.",
    R"json({
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "description": "List of steps with their current status",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": "Short description of the step"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Current status: pending, in_progress, or completed"
                        }
                    },
                    "required": ["step", "status"]
                }
            },
            "explanation": {
                "type": "string",
                "description": "Optional note about current progress or next action"
            }
        },
        "required": ["plan"]
    })json",
    plan_execute
};

REGISTER_TOOL(update_plan, plan_tool);
