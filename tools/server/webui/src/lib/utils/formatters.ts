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
