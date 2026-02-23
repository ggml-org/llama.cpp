/**
 * CORS Proxy utility for routing requests through llama-server's CORS proxy.
 */

import { base } from '$app/paths';

/**
 * Build a proxied URL that routes through llama-server's CORS proxy.
 * @param targetUrl - The original URL to proxy
 * @returns URL pointing to the CORS proxy with target encoded
 */
export function buildProxiedUrl(targetUrl: string): URL {
	const proxyPath = `${base}/cors-proxy`;
	const proxyUrl = new URL(proxyPath, window.location.origin);

	proxyUrl.searchParams.set('url', targetUrl);

	return proxyUrl;
}

/**
 * Get a proxied URL string for use in fetch requests.
 * @param targetUrl - The original URL to proxy
 * @returns Proxied URL as string
 */
export function getProxiedUrlString(targetUrl: string): string {
	return buildProxiedUrl(targetUrl).href;
}
