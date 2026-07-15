import { base } from '$app/paths';
import { error } from '@sveltejs/kit';
import { browser } from '$app/environment';
import { AUTHORIZATION_HEADER, BEARER_PREFIX } from '$lib/constants';
import { config } from '$lib/stores/settings.svelte';

/**
 * Validates API key by making a request to the server props endpoint
 * Throws SvelteKit errors for authentication failures or server issues
 */
export async function validateApiKey(fetch: typeof globalThis.fetch): Promise<void> {
	if (!browser) {
		return;
	}

	const apiKey = config().apiKey;

	// No API key configured — server doesn't require auth, skip the request entirely.
	// The /props endpoint is only protected when the server has API keys configured,
	// and in that case the client always has one set (from settings).
	if (!apiKey) {
		return;
	}

	try {
		const headers: Record<string, string> = {
			'Content-Type': 'application/json',
			[AUTHORIZATION_HEADER]: `${BEARER_PREFIX}${apiKey}`
		};

		const response = await fetch(`${base}/props`, { headers });

		if (!response.ok) {
			if (response.status === 401 || response.status === 403) {
				throw error(401, 'Access denied');
			}

			console.warn(`Server responded with status ${response.status} during API key validation`);
			return;
		}
	} catch (err) {
		// If it's already a SvelteKit error, re-throw it
		if (err && typeof err === 'object' && 'status' in err) {
			throw err;
		}

		// Network or other errors
		console.warn('Cannot connect to server for API key validation:', err);
	}
}
