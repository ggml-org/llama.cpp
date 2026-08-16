/// <reference types="@vitest/browser/matchers" />
/// <reference types="@vitest/browser/providers/playwright" />

import { beforeEach, vi } from 'vitest';

// Mock fetch for API calls during client tests.
// In test environment there is no backend server, so we intercept
// the specific endpoints the app uses and return valid mock data.
// The passthrough target is captured once at module load: capturing it
// inside beforeEach grabs the previous test's spy (vi.spyOn returns the
// existing spy), making the default branch recurse on itself as soon as
// a test fetches a URL outside the mocked set.
const originalFetch = globalThis.fetch.bind(globalThis);

beforeEach(() => {
	vi.spyOn(globalThis, 'fetch').mockImplementation(
		async (input: RequestInfo | URL, init?: RequestInit) => {
			const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

			// Mock server props endpoint
			if (url.includes('/server')) {
				return new Response(
					JSON.stringify({
						git_branch: 'test',
						git_commit: 'test',
						mode: 'router',
						version: 'test'
					}),
					{ headers: { 'Content-Type': 'application/json' }, status: 200 }
				);
			}

			// Mock models list endpoint
			if (/\/v1\/models|\/models\b/.test(url)) {
				return new Response(
					JSON.stringify({
						data: [
							{
								created: 0,
								id: 'test-model.gguf',
								in_cache: false,
								meta: {},
								object: 'model',
								owned_by: 'llamacpp',
								path: 'models/test-model.gguf',
								status: { value: 'unloaded' }
							}
						],
						models: [
							{
								details: {},
								model: 'test-model.gguf',
								name: 'Test Model'
							}
						],
						object: 'list'
					}),
					{ headers: { 'Content-Type': 'application/json' }, status: 200 }
				);
			}

			// Mock /props endpoint (used for modalities)
			if (url.includes('/props')) {
				return new Response(
					JSON.stringify({
						default_generation_settings: { n_ctx: 2048 }
					}),
					{ headers: { 'Content-Type': 'application/json' }, status: 200 }
				);
			}

			// Mock /tools endpoint (used for built-in tools list)
			if (url.includes('/tools')) {
				return new Response(JSON.stringify([]), {
					headers: { 'Content-Type': 'application/json' },
					status: 200
				});
			}

			// Default: use real fetch
			return originalFetch(input, init);
		}
	);
});
