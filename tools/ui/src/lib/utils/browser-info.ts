/**
 * Browser fallback for the server's `get_info` tool, offered only when the
 * server does not serve one (llama-server without --agent). It tells the model
 * which OS the browser runs on and that there is no local file or shell access,
 * so it does not plan around tools that are not there.
 *
 * @see server_tool_get_info in tools/server/server-tools.cpp - the served variant
 * @see buildBrowserInfoToolDefinition in constants/browser-info.ts - tool schema sent to the LLM
 */

import { browser } from '$app/environment';
import {
	BROWSER_INFO_NOTE,
	BROWSER_INFO_OS_UA_PATTERNS,
	BROWSER_INFO_OS_UNKNOWN
} from '$lib/constants';
import type { ToolExecutionResult } from '$lib/types';

function detectOs(userAgent: string): string {
	for (const [pattern, os] of BROWSER_INFO_OS_UA_PATTERNS) {
		if (pattern.test(userAgent)) return os;
	}

	return BROWSER_INFO_OS_UNKNOWN;
}

/**
 * Result shape mirrors the server tool's JSON so the `get_info` renderer reads
 * `os` the same way, minus `cwd` - there is no working directory to report.
 */
export function executeBrowserInfoTool(): ToolExecutionResult {
	return {
		content: JSON.stringify({
			note: BROWSER_INFO_NOTE,
			os: browser ? detectOs(navigator.userAgent) : BROWSER_INFO_OS_UNKNOWN
		}),
		isError: false
	};
}
