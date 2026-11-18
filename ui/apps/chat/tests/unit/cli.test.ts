import { createChatServer, DEFAULTS, parseCliArgs, parseServerUrl } from '../../cli/server.js';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
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
		expect(options.server.href).toBe(`${DEFAULTS.server}/`);
	});

	it('reads the documented flags', () => {
		const options = parseCliArgs([
			'--server',
			'http://192.168.1.20:8080',
			'--host',
			'0.0.0.0',
			'--port',
			'3001',
			'--no-open'
		]);

		expect(options).toMatchObject({ host: '0.0.0.0', open: false, port: 3001 });
		expect(options.server.href).toBe('http://192.168.1.20:8080/');
	});

	it('accepts start as a verb that changes nothing', () => {
		expect(parseCliArgs(['start', '--no-open']).open).toBe(false);
	});

	it('refuses unknown verbs, unknown flags and bad ports', () => {
		expect(() => parseCliArgs(['serve'])).toThrow(/unknown command/);
		expect(() => parseCliArgs(['--nope'])).toThrow();
		expect(() => parseCliArgs(['--port', 'abc'])).toThrow(/invalid --port/);
		expect(() => parseCliArgs(['--port', '70000'])).toThrow(/invalid --port/);
	});

	it('takes a server address without a scheme as http', () => {
		expect(parseServerUrl('localhost:8080').href).toBe('http://localhost:8080/');
		expect(() => parseServerUrl('ftp://localhost')).toThrow(/invalid --server/);
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

describe('llama-chat server', () => {
	let dist: string;
	let upstream: http.Server;
	let upstreamUrl: string;
	let chat: http.Server;
	let chatUrl: string;

	const seen: { headers: http.IncomingHttpHeaders; method?: string; url?: string; body: string }[] =
		[];

	beforeAll(async () => {
		dist = await mkdtemp(join(tmpdir(), 'llama-chat-dist-'));
		await mkdir(join(dist, '_app', 'immutable'), { recursive: true });
		await writeFile(join(dist, 'index.html'), '<!doctype html><title>Llama Chat</title>');
		await writeFile(join(dist, 'sw.js'), 'self;');
		await writeFile(join(dist, 'service-worker.js'), 'self;');
		await writeFile(join(dist, '_app', 'immutable', 'app.abc123.js'), 'export {};');

		upstream = http.createServer((req, res) => {
			let body = '';

			req.on('data', (chunk) => (body += chunk));
			req.on('end', () => {
				seen.push({ body, headers: req.headers, method: req.method, url: req.url });
				res.writeHead(200, { 'Content-Type': 'application/json' });
				res.end(JSON.stringify({ path: req.url }));
			});
		});
		upstreamUrl = await listen(upstream);

		chat = createChatServer({ distDir: dist, server: new URL(`${upstreamUrl}/prefix/`) });
		chatUrl = await listen(chat);
	});

	afterAll(async () => {
		await close(chat);
		await close(upstream);
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

	it('forwards API calls to the llama-server as its own page', async () => {
		const res = await fetch(`${chatUrl}/v1/chat/completions?x=1`, {
			body: '{"messages":[]}',
			headers: { 'Content-Type': 'application/json', Origin: chatUrl },
			method: 'POST'
		});

		expect(res.status).toBe(200);
		expect(await res.json()).toEqual({ path: '/prefix/v1/chat/completions?x=1' });

		const request = seen.at(-1)!;

		expect(request.method).toBe('POST');
		expect(request.body).toBe('{"messages":[]}');
		expect(request.headers.origin).toBeUndefined();
		expect(request.headers.host).toBe(new URL(upstreamUrl).host);
	});

	it("passes another page's Origin through, so llama-server can refuse it", async () => {
		await fetch(`${chatUrl}/props`, {
			headers: { Origin: 'http://evil.example', 'Sec-Fetch-Site': 'cross-site' }
		});

		const request = seen.at(-1)!;

		expect(request.headers.origin).toBe('http://evil.example');
		expect(request.headers['sec-fetch-site']).toBe('cross-site');
	});

	it('refuses other host names on loopback, against DNS rebinding', async () => {
		// fetch() sets Host itself, so the request goes out through http.request
		const status = (host: string) =>
			new Promise<number>((resolve, reject) => {
				const { port } = new URL(chatUrl);

				http
					.request({ headers: { host }, host: '127.0.0.1', path: '/props', port }, (res) => {
						res.resume();
						resolve(res.statusCode ?? 0);
					})
					.on('error', reject)
					.end();
			});

		expect(await status('rebind.example:3000')).toBe(403);
		expect(await status('localhost:3000')).toBe(200);
		expect(await status('[::1]:3000')).toBe(200);
	});

	it('does not serve files from outside dist', async () => {
		// fetch() would resolve the dots itself, so the raw path goes out through http.get
		const path = '/..%2f..%2f..%2fetc%2fpasswd';
		const body = await new Promise<string>((resolve, reject) => {
			http
				.get(`${chatUrl}${path}`, (res) => {
					let data = '';

					res.on('data', (chunk) => (data += chunk));
					res.on('end', () => resolve(data));
				})
				.on('error', reject);
		});

		expect(JSON.parse(body)).toEqual({ path: `/prefix${path}` });
	});

	it('answers 502 when the llama-server is down', async () => {
		const down = createChatServer({ distDir: dist, server: new URL('http://127.0.0.1:9') });
		const url = await listen(down);

		try {
			const res = await fetch(`${url}/props`);

			expect(res.status).toBe(502);
			expect((await res.json()).error.message).toMatch(/could not reach the llama-server/);
		} finally {
			await close(down);
		}
	});
});
