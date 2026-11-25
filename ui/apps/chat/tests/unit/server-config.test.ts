import {
	getServerBase,
	loadServerConfig,
	normalizeServerBase,
	serverUrl,
	setServerBase
} from '#lib/utils/server-config.ts';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('$app/paths', () => ({ asset: (path: string) => `./${path}` }));

afterEach(() => {
	setServerBase('');
	vi.unstubAllGlobals();
});

describe('normalizeServerBase', () => {
	it('keeps the origin and a path, without a trailing slash', () => {
		expect(normalizeServerBase('http://localhost:9931')).toBe('http://localhost:9931');
		expect(normalizeServerBase(' http://localhost:9931/ ')).toBe('http://localhost:9931');
		expect(normalizeServerBase('https://llm.example.com/llama/')).toBe(
			'https://llm.example.com/llama'
		);
	});

	it('is empty for anything but an http(s) URL', () => {
		for (const value of ['', '   ', 'localhost:9931', 'ftp://host', 'javascript:alert(1)', 7, null])
			expect(normalizeServerBase(value)).toBe('');
	});
});

describe('serverUrl', () => {
	it("stays on the app's own origin by default", () => {
		expect(serverUrl('/props')).toBe('./props');
	});

	it('goes to the configured server', () => {
		setServerBase('http://localhost:9931/');

		expect(serverUrl('/props')).toBe('http://localhost:9931/props');
		expect(serverUrl('v1/chat/completions')).toBe('http://localhost:9931/v1/chat/completions');
		expect(serverUrl('./props')).toBe('http://localhost:9931/props');
	});
});

describe('loadServerConfig', () => {
	const respond = (body: unknown, ok = true) =>
		vi.stubGlobal(
			'fetch',
			vi.fn(async () => ({ json: async () => body, ok }))
		);

	it('reads the server from config.json, past the HTTP cache', async () => {
		respond({ server: 'http://localhost:9931' });
		await loadServerConfig();

		expect(fetch).toHaveBeenCalledWith('./config.json', { cache: 'no-store' });
		expect(getServerBase()).toBe('http://localhost:9931');
	});

	it("leaves the server on the app's origin for an empty, missing or broken file", async () => {
		respond({});
		await loadServerConfig();
		expect(getServerBase()).toBe('');

		respond({ server: 'http://localhost:9931' }, false);
		await loadServerConfig();
		expect(getServerBase()).toBe('');

		vi.stubGlobal(
			'fetch',
			vi.fn(async () => {
				throw new Error('offline');
			})
		);
		await loadServerConfig();
		expect(getServerBase()).toBe('');
	});
});
