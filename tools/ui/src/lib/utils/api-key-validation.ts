import { error } from '@sveltejs/kit';
import { browser } from '$app/environment';
import { base } from '$app/paths';
import { HEADERS } from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { settingsStore } from '$lib/stores/settings.svelte';

/**
 * Validates API key by making a request to the server props endpoint
 * Throws SvelteKit errors for authentication failures or server issues
 */
export async function validateApiKey(fetch: typeof globalThis.fetch): Promise<void> {
	if (!browser) {
		return;
	}

	const apiKey = settingsStore.config.apiKey;

	try {
		const headers: Record<string, string> = {
			[HEADERS.CONTENT_TYPE]: MimeTypeApplication.JSON
		};

		// Probe /props even without a stored key: on a server started with
		// --api-key the unauthenticated request returns 401 and surfaces the
		// API key splash, which is the onboarding path for entering the key.
		if (apiKey) {
			headers[HEADERS.AUTHORIZATION] = `${HEADERS.BEARER}${apiKey}`;
		}

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
