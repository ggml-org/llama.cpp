import { beforeEach, describe, expect, it } from 'vitest';
import { validateApiKey } from '$lib/utils/api-key-validation';
import { settingsStore } from '$lib/stores/settings.svelte';
import { CONFIG_LOCALSTORAGE_KEY } from '$lib/constants/storage';

function fakeFetch(status: number, capture: { auth?: string | null } = {}) {
	return (async (_url: RequestInfo | URL, init?: RequestInit) => {
		capture.auth = (init?.headers as Record<string, string>)?.['Authorization'] ?? null;
		return new Response(status === 200 ? '{}' : 'Unauthorized', { status });
	}) as typeof globalThis.fetch;
}

const is401 = (err: unknown) =>
	typeof err === 'object' && err !== null && 'status' in err && err.status === 401;

describe('api key validation surfaces the splash', () => {
	beforeEach(() => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
	});

	it('keyed server, no stored key: throws 401 so the splash shows (onboarding)', async () => {
		await expect(validateApiKey(fakeFetch(401))).rejects.toSatisfy(is401);
	});

	it('keyed server, wrong stored key: throws 401 so the splash shows', async () => {
		settingsStore.updateConfig('apiKey', 'wrong-key');
		await expect(validateApiKey(fakeFetch(401))).rejects.toSatisfy(is401);
	});

	it('open server, no stored key: passes silently', async () => {
		await expect(validateApiKey(fakeFetch(200))).resolves.toBeUndefined();
	});

	it('valid stored key: passes and sends the bearer header', async () => {
		settingsStore.updateConfig('apiKey', 'sk-good');
		const capture: { auth?: string | null } = {};
		await expect(validateApiKey(fakeFetch(200, capture))).resolves.toBeUndefined();
		expect(capture.auth).toBe('Bearer sk-good');
	});
});
