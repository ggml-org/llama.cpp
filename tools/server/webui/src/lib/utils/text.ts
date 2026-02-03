/**
 * Returns a shortened preview of the provided content capped at the given length.
 * Appends an ellipsis when the content exceeds the maximum.
 */
export function getPreviewText(content: string, max = 150): string {
	return content.length > max ? content.slice(0, max) + '...' : content;
}
