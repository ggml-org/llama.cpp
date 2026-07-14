import { TWO_PART_PUBLIC_SUFFIXES, WILDCARD_PUBLIC_SUFFIXES } from '$lib/constants';
import { UrlProtocol } from '$lib/enums';

/**
 * Check whether a hostname looks like an IPv4 or IPv6 address.
 */
function isIpAddress(hostname: string): boolean {
	if (hostname.includes(':')) return true;

	if (/^\d{1,3}(\.\d{1,3}){3}$/.test(hostname)) return true;

	return false;
}

/**
 * Extract the registrable root domain from a URL.
 *
 * @example
 *   'mcp.example.com'           -> 'example.com'
 *   'www.example.co.uk'         -> 'example.co.uk'
 *   'bar.foo.nom.br'            -> 'bar.foo.nom.br'
 *   '192.168.1.1'               -> null
 *   'localhost'                 -> null
 */
export function extractRootDomain(url: URL): string | null {
	const hostname = url.hostname.toLowerCase();
	if (!hostname || isIpAddress(hostname)) return null;

	const parts = hostname.split('.');

	if (parts.length < 2) return null;

	if (parts.length >= 3) {
		const suffix2 = `${parts[parts.length - 2]}.${parts[parts.length - 1]}`;

		if (TWO_PART_PUBLIC_SUFFIXES.has(suffix2)) {
			return parts.slice(-3).join('.');
		}
	}

	for (let i = 2; i <= parts.length; i++) {
		const candidate = parts.slice(-i).join('.');

		if (WILDCARD_PUBLIC_SUFFIXES.has(candidate)) {
			if (parts.length === i + 1) {
				return hostname;
			}

			return parts.slice(-(i + 2)).join('.');
		}
	}

	return parts.slice(-2).join('.');
}

/**
 * Sanitize an external URL string for safe use in an `<a href>`.
 * Only allows http: and https: schemes. Returns `null` for anything else.
 */
export function sanitizeExternalUrl(raw: string): string | null {
	try {
		const url = new URL(raw);

		if (url.protocol !== UrlProtocol.HTTP && url.protocol !== UrlProtocol.HTTPS) {
			return null;
		}

		return url.href;
	} catch {
		return null;
	}
}

/**
 * Canonicalize a server URL for "is this the same server?" checks across
 * the user's settings and the recommended-server list. Lowercases scheme
 * and host, drops default ports, and strips any trailing slashes off the
 * path so a stored `https://api.example.com/mcp/` matches the recommended
 * `https://api.example.com/mcp`. Falls back to a cheap trim+lowercase+strip
 * pass when the input isn't a parseable URL.
 *
 * Query strings are preserved deliberately - if the user entered one,
 * it's part of their endpoint.
 */
export function canonicalizeServerUrl(raw: string): string {
	const trimmed = raw.trim();

	try {
		const parsed = new URL(trimmed);
		const pathname = parsed.pathname.replace(/\/+$/, '');

		return `${parsed.protocol}//${parsed.host}${pathname}${parsed.search}`;
	} catch {
		return trimmed.toLowerCase().replace(/\/+$/, '');
	}
}
