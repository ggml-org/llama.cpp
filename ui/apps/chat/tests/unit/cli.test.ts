import { buildStatic, createChatServer, DEFAULTS, parseCliArgs } from '../../cli/server.js';
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

function listen(server: http.Server): Promise<string> {
	return new Promise((resolve) => {
		server.listen(0, '127.0.0.1', () => {
			resolve(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
		});
	});
}

function close(server: http.Server): Promise<void> {
	return new Promise((resolve) => server.close(() => resolve()));
}

describe('llama-chat CLI arguments', () => {
	it('starts on the defaults with no arguments', () => {
		const options = parseCliArgs([]);

		expect(options.host).toBe(DEFAULTS.host);
		expect(options.port).toBe(DEFAULTS.port);
		expect(options.open).toBe(true);
	});

	it('reads the documented flags', () => {
		const options = parseCliArgs(['--host', '0.0.0.0', '--port', '3001', '--no-open']);

		expect(options).toMatchObject({ host: '0.0.0.0', open: false, port: 3001 });
	});

	it('accepts start as a verb that changes nothing', () => {
		expect(parseCliArgs(['start', '--no-open']).open).toBe(false);
	});

	it('refuses unknown verbs, unknown flags and bad ports', () => {
		expect(() => parseCliArgs(['serve'])).toThrow(/unknown command/);
		expect(() => parseCliArgs(['--nope'])).toThrow();
		expect(() => parseCliArgs(['--port', 'abc'])).toThrow(/invalid --port/);
		expect(() => parseCliArgs(['--port', '70000'])).toThrow(/invalid --port/);
		// it forwards nothing, so there is no llama-server to name
		expect(() => parseCliArgs(['--server', 'http://127.0.0.1:9931'])).toThrow();
	});

	it('reads build with its own flags, and keeps serving flags out of it', () => {
		expect(parseCliArgs(['build'])).toMatchObject({
			command: 'build',
			force: false,
			out: DEFAULTS.out
		});
		expect(parseCliArgs(['build', '--out', 'public', '--force'])).toMatchObject({
			command: 'build',
			force: true,
			out: 'public'
		});
		expect(parseCliArgs([]).command).toBe('serve');
		expect(() => parseCliArgs(['build', '--port', '3001'])).toThrow(/does not apply to build/);
		expect(() => parseCliArgs(['--out', 'public'])).toThrow(/does not apply to serving/);
	});
});

describe('llama-chat packages', () => {
	it('keeps the llama-chat alias on the version of @ggml-org/llama-chat', async () => {
		const read = async (path: string) =>
			JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
		const app = await read('../../package.json');
		const alias = await read('../../alias/package.json');

		expect(app.private).toBeUndefined();
		expect(app.bin['llama-chat']).toBe('./bin/cli.js');
		expect(alias.version).toBe(app.version);
		expect(alias.dependencies[app.name]).toBe(app.version);
	});
});

describe('llama-chat build', () => {
	let dist: string;
	let work: string;

	beforeAll(async () => {
		dist = await mkdtemp(join(tmpdir(), 'llama-chat-dist-'));
		work = await mkdtemp(join(tmpdir(), 'llama-chat-out-'));
		await mkdir(join(dist, '_app'), { recursive: true });
		await writeFile(join(dist, 'index.html'), '<!doctype html>');
		await writeFile(join(dist, '_app', 'app.js'), 'export {};');
		await writeFile(join(dist, '_app', 'app.js.map'), '{}');
	});

	afterAll(async () => {
		await rm(dist, { force: true, recursive: true });
		await rm(work, { force: true, recursive: true });
	});

	it('writes dist to the folder without source maps', async () => {
		const out = await buildStatic({ distDir: dist, out: join(work, 'site') });

		expect((await readdir(out, { recursive: true })).sort()).toEqual([
			'_app',
			join('_app', 'app.js'),
			'index.html'
		]);
	});

	it('refuses a folder that is not empty unless forced', async () => {
		const out = join(work, 'busy');

		await mkdir(out);
		await writeFile(join(out, 'keep.txt'), 'mine');

		await expect(buildStatic({ distDir: dist, out })).rejects.toThrow(/not empty/);
		await buildStatic({ distDir: dist, force: true, out });
		expect(await readFile(join(out, 'keep.txt'), 'utf8')).toBe('mine');
		expect(await readFile(join(out, 'index.html'), 'utf8')).toBe('<!doctype html>');
	});

	it('refuses to write into the app files themselves', async () => {
		await expect(buildStatic({ distDir: dist, force: true, out: dist })).rejects.toThrow(
			/inside the app's own files/
		);
		await expect(
			buildStatic({ distDir: dist, force: true, out: join(dist, '_app', 'copy') })
		).rejects.toThrow(/inside the app's own files/);
	});
});

describe('llama-chat server', () => {
	let dist: string;
	let chat: http.Server;
	let chatUrl: string;

	beforeAll(async () => {
		dist = await mkdtemp(join(tmpdir(), 'llama-chat-dist-'));
		await mkdir(join(dist, '_app', 'immutable'), { recursive: true });
		await writeFile(join(dist, 'index.html'), '<!doctype html><title>Llama Chat</title>');
		await writeFile(join(dist, 'sw.js'), 'self;');
		await writeFile(join(dist, 'service-worker.js'), 'self;');
		await writeFile(join(dist, '_app', 'immutable', 'app.abc123.js'), 'export {};');

		chat = createChatServer({ distDir: dist });
		chatUrl = await listen(chat);
	});

	afterAll(async () => {
		await close(chat);
		await rm(dist, { force: true, recursive: true });
	});

	it('serves index.html at the root, revalidated and cross-origin isolated', async () => {
		const res = await fetch(`${chatUrl}/`);

		expect(res.status).toBe(200);
		expect(res.headers.get('content-type')).toMatch(/^text\/html/);
		expect(res.headers.get('cache-control')).toBe('no-cache');
		expect(res.headers.get('cross-origin-embedder-policy')).toBe('require-corp');
		expect(res.headers.get('cross-origin-opener-policy')).toBe('same-origin');
		expect(await res.text()).toContain('Llama Chat');

		const again = await fetch(`${chatUrl}/`, {
			headers: { 'If-None-Match': res.headers.get('etag') ?? '' }
		});

		expect(again.status).toBe(304);
	});

	it('caches hashed assets for good and revalidates the service worker', async () => {
		const asset = await fetch(`${chatUrl}/_app/immutable/app.abc123.js`);

		expect(asset.headers.get('cache-control')).toContain('immutable');
		expect(asset.headers.get('content-type')).toMatch(/^text\/javascript/);

		const sw = await fetch(`${chatUrl}/sw.js`);

		expect(sw.headers.get('cache-control')).toBe('no-cache');

		const worker = await fetch(`${chatUrl}/service-worker.js`);

		expect(worker.headers.get('cache-control')).toBe('no-cache');
	});

	it('forwards nothing: API paths are not found and other methods are refused', async () => {
		const props = await fetch(`${chatUrl}/props`);

		expect(props.status).toBe(404);

		const post = await fetch(`${chatUrl}/v1/chat/completions`, {
			body: '{"messages":[]}',
			headers: { 'Content-Type': 'application/json' },
			method: 'POST'
		});

		expect(post.status).toBe(405);
		expect(post.headers.get('allow')).toBe('GET, HEAD');
	});

	it('does not serve files from outside dist', async () => {
		// fetch() would resolve the dots itself, so the raw path goes out through http.get
		const status = await new Promise<number>((resolve, reject) => {
			http
				.get(`${chatUrl}/..%2f..%2f..%2fetc%2fpasswd`, (res) => {
					res.resume();
					resolve(res.statusCode ?? 0);
				})
				.on('error', reject);
		});

		expect(status).toBe(404);
	});
});
