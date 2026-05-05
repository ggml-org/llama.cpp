/**
 * Extract root domain from a URL by taking the last two hostname segments.
 * e.g. 'mcp.exa.ai' -> 'exa.ai', 'www.example.com' -> 'example.com'
 */
export function extractRootDomain(url: URL): string | null {
	const parts = url.hostname.split('.');

	if (parts.length < 2) return null;

	return parts.slice(-2).join('.');
}
