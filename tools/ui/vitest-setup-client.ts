/// <reference types="@vitest/browser/matchers" />
/// <reference types="@vitest/browser/providers/playwright" />

import { beforeEach, vi } from 'vitest';

// Mock fetch for API calls during client tests.
// In test environment there is no backend server, so we intercept
// the specific endpoints the app uses and return valid mock data.
beforeEach(() => {
	const originalFetch = globalThis.fetch;

	vi.spyOn(globalThis, 'fetch').mockImplementation(
		async (input: RequestInfo | URL, init?: RequestInit) => {
			const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

			// Mock server props endpoint
			if (url.includes('/server')) {
				return new Response(
					JSON.stringify({
						mode: 'router',
						version: 'test',
						git_commit: 'test',
						git_branch: 'test'
					}),
					{ status: 200, headers: { 'Content-Type': 'application/json' } }
				);
			}

			// Mock models list endpoint
			if (/\/v1\/models|\/models\b/.test(url)) {
				return new Response(
					JSON.stringify({
						object: 'list',
						data: [
							{
								id: 'test-model.gguf',
								object: 'model',
								owned_by: 'llamacpp',
								created: 0,
								in_cache: false,
								path: 'models/test-model.gguf',
								status: { value: 'unloaded' },
								meta: {}
							}
						],
						models: [
							{
								model: 'test-model.gguf',
								name: 'Test Model',
								details: {}
							}
						]
					}),
					{ status: 200, headers: { 'Content-Type': 'application/json' } }
				);
			}

			// Mock /props endpoint (used for modalities)
			if (url.includes('/props')) {
				return new Response(
					JSON.stringify({
						default_generation_settings: { n_ctx: 2048 }
					}),
					{ status: 200, headers: { 'Content-Type': 'application/json' } }
				);
			}

			// Mock /tools endpoint (used for built-in tools list)
			if (url.includes('/tools')) {
				return new Response(JSON.stringify([]), {
					status: 200,
					headers: { 'Content-Type': 'application/json' }
				});
			}

			// Default: use real fetch
			return originalFetch(input, init);
		}
	);
});
