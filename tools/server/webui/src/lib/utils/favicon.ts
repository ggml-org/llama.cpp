/**
 * Favicon utility functions for extracting favicons from URLs.
 */

import { getProxiedUrlString } from './cors-proxy';
import {
	GOOGLE_FAVICON_BASE_URL,
	DEFAULT_FAVICON_SIZE,
	DOMAIN_SEPARATOR,
	ROOT_DOMAIN_MIN_PARTS
} from '$lib/constants';

/**
 * Gets a favicon URL for a given URL using Google's favicon service.
 * Returns null if the URL is invalid.
 *
 * @param urlString - The URL to get the favicon for
 * @returns The favicon URL or null if invalid
 */
export function getFaviconUrl(urlString: string, useProxy = true): string | null {
	try {
		const url = new URL(urlString);
		const hostnameParts = url.hostname.split(DOMAIN_SEPARATOR);
		const rootDomain =
			hostnameParts.length >= ROOT_DOMAIN_MIN_PARTS
				? hostnameParts.slice(-ROOT_DOMAIN_MIN_PARTS).join(DOMAIN_SEPARATOR)
				: url.hostname;

		const googleFaviconUrl = `${GOOGLE_FAVICON_BASE_URL}?domain=${rootDomain}&sz=${DEFAULT_FAVICON_SIZE}`;
		return useProxy ? getProxiedUrlString(googleFaviconUrl) : googleFaviconUrl;
	} catch {
		return null;
	}
}
