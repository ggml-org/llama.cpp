import { API_CACHING_PATTERNS } from '$lib/constants';
import { describe, expect, it } from 'vitest';

/**
 * Regression test for issue #28444 ("Llama.cpp localhost shows the ui even
 * when llama.cpp isnt running").
 *
 * The service worker uses NetworkFirst runtime caching, which falls back to a
 * cached response when the network request fails. If the server liveness /
 * status endpoints (/health, /props) are cache-eligible, a stopped llama.cpp
 * server still yields a cached "server is up" response, so the web UI appears
 * connected to a running server. These endpoints must therefore always be
 * resolved from a live request and never matched by a caching route.
 */
describe('PWA runtime caching does not mask a stopped server', () => {
	const LIVENESS_ENDPOINTS = ['/health', '/props'];
	const CACHEABLE_DATA_ENDPOINTS = ['/models', '/slots', '/tools', '/cors-proxy'];
	const patterns = Object.values(API_CACHING_PATTERNS);
	const isCacheEligible = (path: string) => patterns.some((pattern) => pattern.test(path));

	it('never caches server liveness/status endpoints (/health, /props)', () => {
		for (const endpoint of LIVENESS_ENDPOINTS) {
			const eligible = isCacheEligible(endpoint);

			console.info(`[#28444] ${endpoint} cache-eligible: ${eligible} (expected false)`);
			expect(eligible, `${endpoint} must always hit the network, never a cache`).toBe(false);
		}
	});

	it('still caches non-liveness data endpoints', () => {
		for (const endpoint of CACHEABLE_DATA_ENDPOINTS) {
			const eligible = isCacheEligible(endpoint);

			console.info(`[#28444] ${endpoint} cache-eligible: ${eligible} (expected true)`);
			expect(eligible, `${endpoint} should remain cache-eligible`).toBe(true);
		}
	});
});
