import {
	ensureDataset,
	HF_REGULAR_FILE_MAX_BYTES,
	HfNotWritableError,
	listDatasets,
	uploadTrace,
	whoAmI
} from '../src/index.js';
import {
	HF_MS_PER_SECOND,
	HF_REQUEST_TIMEOUT_MS,
	HF_UPLOAD_MIN_BYTES_PER_SECOND
} from '../src/constants.js';
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
		vi.fn(async (input: string | URL, init?: RequestInit) => {
			const url = String(input);

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

describe('Hub calls', () => {
	it('reads the account and its token role', async () => {
		respond({
			'/api/whoami-v2': () =>
				json({ auth: { accessToken: { role: 'write' } }, name: 'me', orgs: [{ name: 'org' }] })
		});

		const user = await whoAmI(TOKEN);

		expect(user).toMatchObject({ name: 'me', tokenRole: 'write' });
		expect(new Headers(calls[0].init?.headers).get('Authorization')).toBe(`Bearer ${TOKEN}`);
	});

	it('knows no account for a token the Hub rejects', async () => {
		respond({ '/api/whoami-v2': () => json({ error: 'Invalid credentials' }, 401) });

		expect(await whoAmI(TOKEN)).toBeNull();
	});

	it('passes on other Hub errors', async () => {
		respond({ '/api/whoami-v2': () => json({ error: 'Rate limited' }, 429) });

		await expect(whoAmI(TOKEN)).rejects.toThrow('Rate limited');
	});

	it('lists the account datasets, newest first, private ones marked', async () => {
		respond({
			'/api/datasets?direction=-1&limit=100&sort=lastModified&author=me': () =>
				json([
					{ id: 'me/llama-chat-traces', private: false },
					{ id: 'me/secret', private: true }
				])
		});

		const datasets = await listDatasets(TOKEN, 'me');

		expect(datasets).toEqual([
			{ id: 'me/llama-chat-traces', isPrivate: false },
			{ id: 'me/secret', isPrivate: true }
		]);
		expect(new Headers(calls[0].init?.headers).get('Authorization')).toBe(`Bearer ${TOKEN}`);
	});

	it('creates a missing dataset, then commits the trace as one regular file', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({ error: 'not found' }, 404),
			'/api/datasets/me/llama-chat-traces/commit/main': () => json({ commitUrl: 'x' }),
			'/api/datasets/me/llama-chat-traces/preupload/main': () =>
				json({ files: [{ path: 'trace.jsonl', uploadMode: 'regular' }] }),
			'/api/repos/create': () => json({ url: 'x' })
		});

		const url = await uploadTrace(TOKEN, upload);

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

		await uploadTrace(TOKEN, upload);

		expect(calls.some((call) => call.url.endsWith('/api/repos/create'))).toBe(false);
	});

	it('refuses a dataset that stores .jsonl with LFS', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({}),
			'/preupload/main': () => json({ files: [{ uploadMode: 'lfs' }] })
		});

		await expect(uploadTrace(TOKEN, upload)).rejects.toThrow('LFS');
	});

	it('explains a dataset the sign-in may not write to', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({}),
			'/preupload/main': () => json({ error: 'Forbidden' }, 403)
		});

		await expect(uploadTrace(TOKEN, upload)).rejects.toThrow('datasets it created');
		await expect(uploadTrace(TOKEN, upload)).rejects.toMatchObject({
			name: 'HfNotWritableError',
			repoId: 'me/llama-chat-traces'
		});
		await expect(uploadTrace(TOKEN, upload)).rejects.toBeInstanceOf(HfNotWritableError);
	});

	it('explains a dataset the sign-in cannot see, which it cannot create either', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({ error: 'Not found' }, 404),
			'/api/repos/create': () => json({ error: 'You already created this dataset repo' }, 409)
		});

		await expect(uploadTrace(TOKEN, upload)).rejects.toThrow(
			'You cannot add traces to me/llama-chat-traces.'
		);
	});

	it('refuses a trace too large for a regular commit before any request', async () => {
		respond({});

		await expect(
			uploadTrace(TOKEN, {
				...upload,
				content: 'x'.repeat(HF_REGULAR_FILE_MAX_BYTES + 1)
			})
		).rejects.toThrow('too large');
		expect(calls).toHaveLength(0);
	});

	it('gives up on a call the Hub does not answer, instead of waiting for good', async () => {
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({}),
			'/preupload/main': () => {
				throw new DOMException('The operation timed out.', 'TimeoutError');
			}
		});

		await expect(uploadTrace(TOKEN, upload)).rejects.toThrow('did not answer in time');
		// every call carries the time limit
		expect(calls.every((call) => call.init?.signal instanceof AbortSignal)).toBe(true);
	});

	it('refuses a dataset name that is not owner/name before any request', async () => {
		respond({});

		await expect(uploadTrace(TOKEN, { ...upload, repoId: 'owner/name/extra' })).rejects.toThrow(
			'owner/name'
		);
		await expect(ensureDataset(TOKEN, 'name', false)).rejects.toThrow('owner/name');
		expect(calls).toHaveLength(0);
	});

	it('gives a large body the time to go out, on top of the usual limit', async () => {
		const timeouts: number[] = [];
		const timeout = AbortSignal.timeout.bind(AbortSignal);

		vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms: number) => {
			timeouts.push(ms);

			return timeout(ms);
		});
		respond({
			'/api/datasets/me/llama-chat-traces': () => json({}),
			'/commit/main': () => json({}),
			'/preupload/main': () => json({ files: [{ uploadMode: 'regular' }] })
		});

		const content = 'x'.repeat(4 * HF_UPLOAD_MIN_BYTES_PER_SECOND);

		await uploadTrace(TOKEN, { ...upload, content });

		// the lookup and the check carry little; the commit carries the file, base64 encoded
		const encoded = Math.ceil((content.length * 4) / 3 / HF_UPLOAD_MIN_BYTES_PER_SECOND);

		expect(timeouts[0]).toBe(HF_REQUEST_TIMEOUT_MS);
		expect(timeouts.at(-1)).toBeGreaterThanOrEqual(
			HF_REQUEST_TIMEOUT_MS + encoded * HF_MS_PER_SECOND
		);
	});
});
