/**
 * Browser executor for the `get_datetime` tool. It runs in the browser, so it
 * reports the user's own clock and time zone instead of the server's UTC time -
 * a chat about "tomorrow" means the user's tomorrow, not the host's.
 *
 * @see buildGetDatetimeToolDefinition in constants/get-datetime.ts - tool schema sent to the LLM
 */

import type { ToolExecutionResult } from '$lib/types';

function pad(value: number): string {
	return String(value).padStart(2, '0');
}

/** ISO 8601 in local time, e.g. `2026-08-17T14:05:09+02:00` */
function localIsoString(date: Date): string {
	// getTimezoneOffset() counts minutes behind UTC, ISO 8601 counts them ahead
	const offset = -date.getTimezoneOffset();
	const sign = offset < 0 ? '-' : '+';
	const absOffset = Math.abs(offset);
	const day = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
	const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;

	return `${day}T${time}${sign}${pad(Math.floor(absOffset / 60))}:${pad(absOffset % 60)}`;
}

/** The `result` field keeps the shape the `get_datetime` renderer already reads. */
export function executeGetDatetimeTool(): ToolExecutionResult {
	const now = new Date();

	return {
		content: JSON.stringify({
			result: localIsoString(now),
			timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
		}),
		isError: false
	};
}
