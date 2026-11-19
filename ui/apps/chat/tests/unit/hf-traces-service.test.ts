import { HF_REGULAR_FILE_MAX_BYTES } from '#lib/constants/index.ts';
import { HfTracesService } from '#lib/services/hf-traces.service.ts';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const TOKEN = 'hf_test';

interface Call {
	url: string;
	init?: RequestInit;
}

let calls: Call[];

function respond(routes: Record<string, () => Response>) {
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			calls.push({ init, url });

			const route = Object.keys(routes).find((key) => url.endsWith(key));

			if (!route) throw new Error(`unexpected request: ${url}`);

			return routes[route]();
		})
	);
}

function json(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status });
}

const upload = {
	commitMessage: 'Add trace: test',
	content: '{"type":"session","harness":"llama.app","id":"c1"}',
	isPrivate: true,
	path: 'trace.jsonl',
	repoId: 'me/llama-chat-traces'
};

beforeEach(() => {
	calls = [];
});

afterEach(() => {
	vi.unstubAllGlobals();
});

describe('HfTracesService', () => {
	it('reads the account and its token role', async () => {
		respond({
			'/api/whoami-v2': () =>
				json({ auth: { accessToken: { role: 'write' } }, name: 'me', orgs: [{ name: 'org' }] })
		});

		const user = await HfTracesService.whoAmI(TOKEN);

		expect(user).toMatchObject({ name: 'me', tokenRole: 'write' });
		expect(new Headers(calls[0].init?.headers).get('Authorization')).toBe(`Bearer ${TOKEN}`);
	});

	it('says so when the Hub rejects the token', async () => {
		respond({ '/api/whoami-v2': () => json({ error: 'Invalid credentials' }, 401) });

		await expect(HfTracesService.whoAmI(TOKEN)).rejects.toThrow('did not accept this token');
	});

	it('passes on other Hub errors', async () => {
		respond({ '/api/whoami-v2': () => json({ error: 'Rate limited' }, 429) });

		await expect(HfTracesService.whoAmI(TOKEN)).rejects.toThrow('Rate limited');
	});

	it('creates a missing dataset, then commits the trace as one regular file', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({ error: 'not found' }, 404),
			'/api/datasets/me/llama-chat-traces/commit/main': () => json({ commitUrl: 'x' }),
			'/api/datasets/me/llama-chat-traces/preupload/main': () =>
				json({ files: [{ path: 'trace.jsonl', uploadMode: 'regular' }] }),
			'/api/repos/create': () => json({ url: 'x' })
		});

		const url = await HfTracesService.uploadTrace(TOKEN, upload);

		expect(url).toBe('https://huggingface.co/datasets/me/llama-chat-traces/blob/main/trace.jsonl');
		expect(JSON.parse(String(calls[1].init?.body))).toEqual({
			name: 'llama-chat-traces',
			organization: 'me',
			private: true,
			type: 'dataset'
		});

		const [header, file] = String(calls[3].init?.body)
			.split('\n')
			.map((line) => JSON.parse(line));

		expect(header).toEqual({
			key: 'header',
			value: { description: '', summary: upload.commitMessage }
		});
		expect(file.key).toBe('file');
		expect(file.value.path).toBe('trace.jsonl');
		expect(atob(file.value.content)).toBe(upload.content);
	});

	it('leaves an existing dataset as it is', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({ id: 'me/llama-chat-traces' }),
			'/commit/main': () => json({}),
			'/preupload/main': () => json({ files: [{ uploadMode: 'regular' }] })
		});

		await HfTracesService.uploadTrace(TOKEN, upload);

		expect(calls.some((call) => call.url.endsWith('/api/repos/create'))).toBe(false);
	});

	it('refuses a dataset that stores .jsonl with LFS', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({}),
			'/preupload/main': () => json({ files: [{ uploadMode: 'lfs' }] })
		});

		await expect(HfTracesService.uploadTrace(TOKEN, upload)).rejects.toThrow('LFS');
	});

	it('refuses a trace too large for a regular commit before any request', async () => {
		respond({});

		await expect(
			HfTracesService.uploadTrace(TOKEN, {
				...upload,
				content: 'x'.repeat(HF_REGULAR_FILE_MAX_BYTES + 1)
			})
		).rejects.toThrow('too large');
		expect(calls).toHaveLength(0);
	});
});
