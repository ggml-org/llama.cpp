/**
 * Decode base64 to UTF-8 string using modern APIs.
 * Falls back to legacy method if TextDecoder is unavailable.
 *
 * @param base64 - Base64 encoded string (padding is optional)
 * @returns Decoded UTF-8 string, or empty string if decoding fails
 */
export function decodeBase64(base64: string): string {
	if (!base64) return '';

	const padded = base64 + '=='.slice(0, (4 - (base64.length % 4)) % 4);

	try {
		const binaryString = atob(padded);
		const bytes = new Uint8Array(binaryString.length);
		for (let i = 0; i < binaryString.length; i++) {
			bytes[i] = binaryString.charCodeAt(i);
		}
		return new TextDecoder('utf-8').decode(bytes);
	} catch {
		// Fallback to legacy method
		try {
			return decodeURIComponent(escape(atob(padded)));
		} catch {
			// Return empty string if all decoding fails
			return '';
		}
	}
}
