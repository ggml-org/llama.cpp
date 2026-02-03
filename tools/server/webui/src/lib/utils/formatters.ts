import {
	MS_PER_SECOND,
	SECONDS_PER_MINUTE,
	SECONDS_PER_HOUR,
	SHORT_DURATION_THRESHOLD,
	MEDIUM_DURATION_THRESHOLD
} from '$lib/constants/formatters';

/**
 * Formats file size in bytes to human readable format
 * Supports Bytes, KB, MB, and GB
 *
 * @param bytes - File size in bytes (or unknown for safety)
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number | unknown): string {
	if (typeof bytes !== 'number') return 'Unknown';
	if (bytes === 0) return '0 Bytes';

	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));

	return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format parameter count to human-readable format (B, M, K)
 *
 * @param params - Parameter count
 * @returns Human-readable parameter count
 */
export function formatParameters(params: number | unknown): string {
	if (typeof params !== 'number') return 'Unknown';

	if (params >= 1e9) {
		return `${(params / 1e9).toFixed(2)}B`;
	}

	if (params >= 1e6) {
		return `${(params / 1e6).toFixed(2)}M`;
	}

	if (params >= 1e3) {
		return `${(params / 1e3).toFixed(2)}K`;
	}

	return params.toString();
}

/**
 * Format number with locale-specific thousands separators
 *
 * @param num - Number to format
 * @returns Human-readable number
 */
export function formatNumber(num: number | unknown): string {
	if (typeof num !== 'number') return 'Unknown';

	return num.toLocaleString();
}

/**
 * Format JSON string with pretty printing (2-space indentation)
 * Returns original string if parsing fails
 *
 * @param jsonString - JSON string to format
 * @returns Pretty-printed JSON string or original if invalid
 */
export function formatJsonPretty(jsonString: string): string {
	try {
		const parsed = JSON.parse(jsonString);
		return JSON.stringify(parsed, null, 2);
	} catch {
		return jsonString;
	}
}

/**
 * Format time as HH:MM:SS in 24-hour format
 *
 * @param date - Date object to format
 * @returns Formatted time string (HH:MM:SS)
 */
export function formatTime(date: Date): string {
	return date.toLocaleTimeString('en-US', {
		hour12: false,
		hour: '2-digit',
		minute: '2-digit',
		second: '2-digit'
	});
}

/**
 * Formats milliseconds to a human-readable time string for performance metrics.
 * Examples: "4h 12min 54s", "12min 34s", "45s", "0.5s"
 *
 * @param ms - Time in milliseconds
 * @returns Formatted time string
 */
export function formatPerformanceTime(ms: number): string {
	if (ms < 0) return '0s';

	const totalSeconds = ms / MS_PER_SECOND;

	if (totalSeconds < SHORT_DURATION_THRESHOLD) {
		return `${totalSeconds.toFixed(1)}s`;
	}

	if (totalSeconds < MEDIUM_DURATION_THRESHOLD) {
		return `${totalSeconds.toFixed(1)}s`;
	}

	const hours = Math.floor(totalSeconds / SECONDS_PER_HOUR);
	const minutes = Math.floor((totalSeconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
	const seconds = Math.floor(totalSeconds % SECONDS_PER_MINUTE);

	const parts: string[] = [];

	if (hours > 0) {
		parts.push(`${hours}h`);
	}

	if (minutes > 0) {
		parts.push(`${minutes}min`);
	}

	if (seconds > 0 || parts.length === 0) {
		parts.push(`${seconds}s`);
	}

	return parts.join(' ');
}

/**
 * Formats attachment content for API requests with consistent header style.
 * Used when converting message attachments to text content parts.
 *
 * @param label - Type label (e.g., 'File', 'PDF File', 'MCP Prompt')
 * @param name - File or attachment name
 * @param content - The actual content to include
 * @param extra - Optional extra info to append to name (e.g., server name for MCP)
 * @returns Formatted string with header and content
 */
export function formatAttachmentText(
	label: string,
	name: string,
	content: string,
	extra?: string
): string {
	const header = extra ? `${name} (${extra})` : name;
	return `\n\n--- ${label}: ${header} ---\n${content}`;
}
