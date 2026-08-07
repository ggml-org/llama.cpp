/**
 * Creates a base64 data URL from MIME type and base64-encoded data.
 *
 * @param mimeType - The MIME type (e.g., 'image/png', 'audio/mp3')
 * @param base64Data - The base64-encoded data
 * @returns A data URL string in format 'data:{mimeType};base64,{data}'
 */
export function createBase64DataUrl(mimeType: string, base64Data: string): string {
	return `data:${mimeType};base64,${base64Data}`;
}
