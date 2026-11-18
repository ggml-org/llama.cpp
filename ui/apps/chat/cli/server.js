/**
 * `npx @ggml-org/llama-chat`: serves the prebuilt Llama Chat (`dist`) on localhost and forwards
 * its API calls to a llama-server. Node built-ins only, so the published package has no
 * dependencies: no llama-server, no models, no Vite or SvelteKit.
 *
 * The app calls the server on its own origin (`./v1/...`, `./props`, ...), as it does when
 * llama-server serves it. So a request is answered from `dist` when a file there matches its
 * path, and forwarded to the llama-server otherwise.
 */

import { spawn } from 'node:child_process';
import { createReadStream, readFileSync } from 'node:fs';
import { stat } from 'node:fs/promises';
import http from 'node:http';
import https from 'node:https';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

export const PACKAGE_ROOT = fileURLToPath(new URL('..', import.meta.url));
export const DIST_DIR = join(PACKAGE_ROOT, 'dist');

export const DEFAULTS = Object.freeze({
	host: '127.0.0.1',
	// fixed, so the origin, and the chats the browser stores under it, stay the same between runs
	port: 3000,
	// where llama-server listens when started without --host or --port
	server: 'http://127.0.0.1:8080'
});

// The same caching llama-server applies to the UI it embeds (tools/server/server-http.cpp):
// index.html and the PWA's update checks are revalidated, hashed assets are cached for good.
const NO_CACHE = new Set([
	'/index.html',
	'/service-worker.js',
	'/sw.js',
	'/manifest.webmanifest',
	'/_app/version.json',
	'/build.json'
]);
const CACHE_IMMUTABLE = 'public, max-age=31536000, immutable';
const CACHE_REVALIDATE = 'no-cache';
/** @type {Record<string, string>} */
const MIME_TYPES = {
	'.css': 'text/css; charset=utf-8',
	'.gif': 'image/gif',
	'.html': 'text/html; charset=utf-8',
	'.ico': 'image/x-icon',
	'.jpg': 'image/jpeg',
	'.js': 'text/javascript; charset=utf-8',
	'.json': 'application/json; charset=utf-8',
	'.map': 'application/json; charset=utf-8',
	'.mjs': 'text/javascript; charset=utf-8',
	'.png': 'image/png',
	'.svg': 'image/svg+xml',
	'.ttf': 'font/ttf',
	'.txt': 'text/plain; charset=utf-8',
	'.wasm': 'application/wasm',
	'.webmanifest': 'application/manifest+json; charset=utf-8',
	'.webp': 'image/webp',
	'.woff': 'font/woff',
	'.woff2': 'font/woff2'
};
// hop-by-hop headers, which a proxy does not pass on
const HOP_BY_HOP = [
	'connection',
	'keep-alive',
	'proxy-authenticate',
	'proxy-authorization',
	'proxy-connection',
	'te',
	'trailer',
	'transfer-encoding',
	'upgrade'
];

export const USAGE = `Usage: llama-chat [options]

Serves Llama Chat on localhost and connects it to a llama-server.

Options:
  --server <url>   llama-server to connect to (default: ${DEFAULTS.server})
  --port <n>       port to listen on (default: ${DEFAULTS.port})
  --host <addr>    address to listen on, 0.0.0.0 for the LAN (default: ${DEFAULTS.host})
  --open           open the browser once it listens (default)
  --no-open        do not open the browser
  -v, --version    print the version
  -h, --help       print this help

Examples:
  npx @ggml-org/llama-chat
  npx @ggml-org/llama-chat --server http://192.168.1.20:8080
  npx @ggml-org/llama-chat --host 0.0.0.0 --port 3000 --no-open`;

/** The version in this package's package.json. */
export function packageVersion() {
	return JSON.parse(readFileSync(join(PACKAGE_ROOT, 'package.json'), 'utf8')).version;
}

/**
 * Reads the command line. `start` is accepted as a verb and means the same as none.
 * @param {string[]} argv the arguments after the command
 * @returns {{ help: boolean, host: string, open: boolean, port: number, server: URL, version: boolean }}
 */
export function parseCliArgs(argv) {
	const { positionals, values } = parseArgs({
		allowNegative: true,
		allowPositionals: true,
		args: argv,
		options: {
			help: { short: 'h', type: 'boolean' },
			host: { type: 'string' },
			open: { type: 'boolean' },
			port: { type: 'string' },
			server: { type: 'string' },
			version: { short: 'v', type: 'boolean' }
		},
		strict: true
	});
	const [verb, ...rest] = positionals;

	if ((verb !== undefined && verb !== 'start') || rest.length > 0) {
		throw new Error(`unknown command: ${positionals.join(' ')}`);
	}

	const port = Number(values.port ?? DEFAULTS.port);

	if (!Number.isInteger(port) || port < 0 || port > 65535) {
		throw new Error(`invalid --port: ${values.port}`);
	}

	return {
		help: values.help ?? false,
		host: values.host ?? DEFAULTS.host,
		open: values.open ?? true,
		port,
		server: parseServerUrl(values.server ?? DEFAULTS.server),
		version: values.version ?? false
	};
}

/**
 * @param {string} value a llama-server address, with or without `http://`
 * @returns {URL}
 */
export function parseServerUrl(value) {
	const url = new URL(/^[a-z][a-z\d+.-]*:\/\//i.test(value) ? value : `http://${value}`);

	if (url.protocol !== 'http:' && url.protocol !== 'https:') {
		throw new Error(`invalid --server: ${value} (use an http:// or https:// address)`);
	}

	return url;
}

/**
 * Maps a request path to a file under `root`, or null when it falls outside it.
 * @param {string} root
 * @param {string} pathname
 */
function resolveStatic(root, pathname) {
	let decoded;

	try {
		decoded = decodeURIComponent(pathname);
	} catch {
		return null;
	}

	if (decoded.includes('\0')) return null;

	const file = normalize(join(root, decoded === '/' ? '/index.html' : decoded));

	return file === root || file.startsWith(root.endsWith(sep) ? root : root + sep) ? file : null;
}

/**
 * Sends a file from `dist`, with llama-server's caching and cross-origin isolation headers.
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {string} file
 * @param {string} pathname
 * @param {import('node:fs').Stats} stats
 */
function sendFile(req, res, file, pathname, stats) {
	const name = pathname === '/' ? '/index.html' : pathname;
	const etag = `W/"${stats.size.toString(16)}-${Math.floor(stats.mtimeMs).toString(16)}"`;

	res.setHeader('ETag', etag);
	res.setHeader('Cache-Control', NO_CACHE.has(name) ? CACHE_REVALIDATE : CACHE_IMMUTABLE);

	if (name === '/index.html') {
		res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
		res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
	}

	if (req.headers['if-none-match'] === etag) {
		res.writeHead(304).end();

		return;
	}

	res.setHeader(
		'Content-Type',
		MIME_TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream'
	);
	res.setHeader('Content-Length', stats.size);
	res.writeHead(200);

	if (req.method === 'HEAD') {
		res.end();

		return;
	}

	createReadStream(file).pipe(res);
}

/**
 * The host name in a Host header, lower case and without the port or IPv6 brackets.
 * @param {string} host
 */
function hostName(host) {
	const name = host.startsWith('[') ? host.slice(1, host.indexOf(']')) : host.split(':')[0];

	return name.toLowerCase();
}

/** @param {string} address */
function isLoopback(address) {
	return /^(127\.|::1$|::ffff:127\.)/.test(address);
}

/**
 * DNS rebinding guard, as llama-server has: a page whose own name resolves to 127.0.0.1 reaches
 * a loopback listener with that name in Host, so there Host must name loopback itself.
 * @param {http.IncomingMessage} req
 */
function hostIsAllowed(req) {
	if (!isLoopback(req.socket.localAddress ?? '')) return true;

	const name = hostName(req.headers.host ?? '');

	return name === 'localhost' || name === '0.0.0.0' || name === '::' || isLoopback(name);
}

/**
 * Forwards a request to the llama-server. llama-server takes cross-origin browser requests only
 * from its --cors-origins, so, like the Vite dev proxy, this calls it as its own page: with the
 * Host of the server, and without the Origin of this page. Another page's Origin and
 * Sec-Fetch-Site pass through, so llama-server still refuses what it would refuse directly.
 * @param {http.IncomingMessage} req
 * @param {http.ServerResponse} res
 * @param {URL} server
 */
function proxy(req, res, server) {
	const prefix = server.pathname.replace(/\/+$/, '');
	const target = new URL(prefix + (req.url ?? '/'), server.origin);
	/** @type {http.OutgoingHttpHeaders} */
	const headers = { ...req.headers, host: target.host };
	const origin = req.headers.origin ?? '';

	if (origin.slice(origin.indexOf('://') + 3).toLowerCase() === req.headers.host?.toLowerCase()) {
		delete headers.origin;
	}

	for (const name of HOP_BY_HOP) delete headers[name];

	const upstream = (target.protocol === 'https:' ? https : http).request(
		target,
		{ headers, method: req.method },
		(response) => {
			const responseHeaders = { ...response.headers };

			for (const name of HOP_BY_HOP) delete responseHeaders[name];

			res.writeHead(response.statusCode ?? 502, responseHeaders);
			response.pipe(res);
		}
	);

	upstream.on('error', (error) => {
		if (res.headersSent) {
			res.destroy(error);

			return;
		}

		res.writeHead(502, { 'Content-Type': 'application/json; charset=utf-8' });
		res.end(
			JSON.stringify({
				error: {
					code: 502,
					message: `llama-chat could not reach the llama-server at ${server.href}: ${error.message}`,
					type: 'unavailable_error'
				}
			})
		);
	});

	// a stream the browser stops reading (Stop, a closed tab) stops the generation on the server
	res.on('close', () => {
		if (!res.writableFinished) upstream.destroy();
	});

	req.pipe(upstream);
}

/**
 * The HTTP server: files from `dist`, everything else from the llama-server.
 * @param {{ distDir?: string, server: URL }} options
 */
export function createChatServer({ distDir = DIST_DIR, server }) {
	const root = normalize(distDir);

	return http.createServer(async (req, res) => {
		const { pathname } = new URL(req.url ?? '/', 'http://localhost');

		if (!hostIsAllowed(req)) {
			res.writeHead(403, { 'Content-Type': 'application/json; charset=utf-8' });
			res.end(
				JSON.stringify({
					error: {
						code: 403,
						message: `llama-chat refused the host name ${req.headers.host}, open it on localhost or 127.0.0.1`,
						type: 'permission_error'
					}
				})
			);

			return;
		}

		if (req.method === 'GET' || req.method === 'HEAD') {
			const file = resolveStatic(root, pathname);
			const stats = file ? await stat(file).catch(() => null) : null;

			if (file && stats?.isFile()) {
				sendFile(req, res, file, pathname, stats);

				return;
			}
		}

		proxy(req, res, server);
	});
}

/**
 * Opens a URL in the default browser, quietly doing nothing where that fails.
 * @param {string} url
 */
function openBrowser(url) {
	const [command, args] =
		process.platform === 'darwin'
			? ['open', [url]]
			: process.platform === 'win32'
				? ['cmd', ['/c', 'start', '""', url]]
				: ['xdg-open', [url]];

	try {
		// verbatim, or Node quotes the empty title of `start` into one cmd.exe misreads
		spawn(command, args, { detached: true, stdio: 'ignore', windowsVerbatimArguments: true })
			.on('error', () => {})
			.unref();
	} catch {
		// no browser to open, e.g. on a headless machine
	}
}

/**
 * Runs the command.
 * @param {string[]} argv the arguments after the command
 * @returns {Promise<http.Server | undefined>} the server, once it listens
 */
export async function main(argv) {
	let options;

	try {
		options = parseCliArgs(argv);
	} catch (error) {
		console.error(`llama-chat: ${error instanceof Error ? error.message : error}\n\n${USAGE}`);
		process.exitCode = 2;

		return;
	}

	if (options.help) {
		console.log(USAGE);

		return;
	}

	if (options.version) {
		console.log(packageVersion());

		return;
	}

	if (!(await stat(join(DIST_DIR, 'index.html')).catch(() => null))) {
		console.error(`llama-chat: no build found in ${DIST_DIR}, run \`npm run build\` first`);
		process.exitCode = 1;

		return;
	}

	const server = createChatServer({ server: options.server });

	return new Promise((resolve) => {
		server.on('error', (error) => {
			const busy = /** @type {NodeJS.ErrnoException} */ (error).code === 'EADDRINUSE';

			console.error(
				busy
					? `llama-chat: port ${options.port} is in use, pick another with --port`
					: `llama-chat: ${error.message}`
			);
			process.exitCode = 1;
			resolve(undefined);
		});

		server.listen(options.port, options.host, () => {
			const address = /** @type {import('node:net').AddressInfo} */ (server.address());
			const shown = ['0.0.0.0', '::'].includes(options.host) ? 'localhost' : options.host;
			const url = `http://${shown.includes(':') ? `[${shown}]` : shown}:${address.port}/`;

			console.log(`Llama Chat ${packageVersion()} on ${url}`);
			console.log(`Connected to the llama-server at ${options.server.href}`);

			if (options.open) openBrowser(url);

			resolve(server);
		});
	});
}
