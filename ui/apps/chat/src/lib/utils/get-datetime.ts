/**
 * Browser executor for the `get_datetime` tool. It runs in the browser, so it
 * reports the user's own clock and time zone instead of the server's UTC time -
 * a chat about "tomorrow" means the user's tomorrow, not the host's.
 *
 * @see buildGetDatetimeToolDefinition in constants/get-datetime.ts - tool schema sent to the LLM
 */

import { formatLocalIso } from './formatters';
import type { ToolExecutionResult } from '$lib/types';

/** The `result` field keeps the shape the `get_datetime` renderer already reads. */
export function executeGetDatetimeTool(): ToolExecutionResult {
	const now = new Date();

	return {
		content: JSON.stringify({
			result: formatLocalIso(now),
			timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
		}),
		isError: false
	};
}
