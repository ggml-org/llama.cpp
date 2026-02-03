/**
 * Favicon utility functions for extracting favicons from URLs.
 */

import { getProxiedUrlString } from './cors-proxy';
import { GOOGLE_FAVICON_BASE_URL, DEFAULT_FAVICON_SIZE } from '$lib/constants/favicon';

/**
 * Gets a favicon URL for a given URL using Google's favicon service.
 * Returns null if the URL is invalid.
 *
 * @param urlString - The URL to get the favicon for
 * @returns The favicon URL or null if invalid
 */
export function getFaviconUrl(urlString: string): string | null {
	try {
		const url = new URL(urlString);
		const hostnameParts = url.hostname.split('.');
		const rootDomain = hostnameParts.length >= 2 ? hostnameParts.slice(-2).join('.') : url.hostname;

		const googleFaviconUrl = `${GOOGLE_FAVICON_BASE_URL}?domain=${rootDomain}&sz=${DEFAULT_FAVICON_SIZE}`;
		return getProxiedUrlString(googleFaviconUrl);
	} catch {
		return null;
	}
}
