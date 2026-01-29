/**
 * Favicon utility functions for extracting favicons from URLs.
 */

import { getProxiedUrlString } from './cors-proxy';

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

		const googleFaviconUrl = `https://www.google.com/s2/favicons?domain=${rootDomain}&sz=32`;
		return getProxiedUrlString(googleFaviconUrl);
	} catch {
		return null;
	}
}
