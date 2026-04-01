/**
 * CORS Proxy utility for routing requests through llama-server's CORS proxy.
 */

import { base } from '$app/paths';
import { CORS_PROXY_ENDPOINT, CORS_PROXY_URL_PARAM } from '$lib/constants';

/**
 * Build a proxied URL that routes through llama-server's CORS proxy.
 * @param targetUrl - The original URL to proxy
 * @returns URL pointing to the CORS proxy with target encoded
 */
export function buildProxiedUrl(targetUrl: string): URL {
	const proxyPath = `${base}${CORS_PROXY_ENDPOINT}`;
	const proxyUrl = new URL(proxyPath, window.location.origin);

	proxyUrl.searchParams.set(CORS_PROXY_URL_PARAM, targetUrl);

	return proxyUrl;
}

/**
 * Wrap original headers for proxying through the CORS proxy. This avoids issues with duplicated llama.cpp-specific and target headers when using the CORS proxy.
 * @param headers - The original headers to be proxied to target
 * @returns List of "wrapped" headers to be sent to the CORS proxy
 */
export function buildProxiedHeaders(headers: Record<string, string>): Record<string, string> {
	const proxiedHeaders: Record<string, string> = {};

	for (const [key, value] of Object.entries(headers)) {
		proxiedHeaders[`x-proxy-header-${key}`] = value;
	}

	return proxiedHeaders;
}

/**
 * Get a proxied URL string for use in fetch requests.
 * @param targetUrl - The original URL to proxy
 * @returns Proxied URL as string
 */
export function getProxiedUrlString(targetUrl: string): string {
	return buildProxiedUrl(targetUrl).href;
}
