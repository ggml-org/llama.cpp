/**
 * `npx @ggml-org/llama-chat`: serves the prebuilt Llama Chat (`dist`) as static files on
 * localhost. `npx @ggml-org/llama-chat build` writes the same files to a folder instead, for any
 * static file server. Node built-ins only, so the published package has no dependencies: no
 * llama-server, no models, no Vite or SvelteKit.
 *
 * It only serves files and forwards nothing: the browser calls each llama-server the app is
 * pointed at directly, so llama-server's own rules (--cors-origins, --api-key) apply as they do
 * to any other page.
 */

import { spawn } from 'node:child_process';
import { createReadStream, readFileSync } from 'node:fs';
import { cp, readdir, stat, writeFile } from 'node:fs/promises';
import http from 'node:http';
import { extname, isAbsolute, join, normalize, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

export const PACKAGE_ROOT = fileURLToPath(new URL('..', import.meta.url));
export const DIST_DIR = join(PACKAGE_ROOT, 'dist');

export const DEFAULTS = Object.freeze({
	host: '127.0.0.1',
	// the folder `build` writes to, in the current directory
	out: 'llama-chat',
	// fixed, so the origin, and the chats the browser stores under it, stay the same between runs
	port: 3000,
	// llama-server's own default port, on the host the app is opened from, when no --server
	serverPort: 9931
});

// The same caching llama-server applies to the UI it embeds (tools/server/server-http.cpp):
// index.html and the PWA's update checks are revalidated, hashed assets are cached for good.
const NO_CACHE = new Set([
	'/index.html',
	'/service-worker.js',
	'/sw.js',
	'/manifest.webmanifest',
	'/_app/version.json',
	'/build.json',
	'/config.json'
]);
// the file next to index.html that tells the app where llama-server is
const CONFIG_PATH = '/config.json';
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
const SERVE_OPTIONS = ['host', 'open', 'port'];
const BUILD_OPTIONS = ['force', 'out'];

export const USAGE = `Usage: llama-chat [options]
       llama-chat build [--out <dir>] [--force]

Serves Llama Chat on localhost as static files, or, with build, writes them to
a folder for any HTTP server. The browser talks to llama-server directly: name
it with --server, and start it with --cors-origins set to this page's address
when it restricts origins.

Options:
  --server <url>   the llama-server the app talks to, e.g. http://localhost:8080
                   (default: port ${DEFAULTS.serverPort} of the host the page is opened on;
                   for build: the address the app is served on)
  --port <n>       port to listen on (default: ${DEFAULTS.port})
  --host <addr>    address to listen on, 0.0.0.0 for the LAN (default: ${DEFAULTS.host})
  --open           open the browser once it listens (default)
  --no-open        do not open the browser
  -v, --version    print the version
  -h, --help       print this help

Build options:
  --out <dir>      folder to write to (default: ./${DEFAULTS.out})
  --force          write into a folder that is not empty

Examples:
  npx @ggml-org/llama-chat
  npx @ggml-org/llama-chat --server http://localhost:8080
  npx @ggml-org/llama-chat --host 0.0.0.0 --port 3000 --no-open
  npx @ggml-org/llama-chat build --out ./public --server https://llm.example.com`;

/** The version in this package's package.json. */
export function packageVersion() {
	return JSON.parse(readFileSync(join(PACKAGE_ROOT, 'package.json'), 'utf8')).version;
}

/**
 * Reads the command line. No verb serves the app, `start` means the same, and `build` writes it
 * to a folder.
 * @param {string[]} argv the arguments after the command
 * @returns {{ command: 'build' | 'serve', force: boolean, help: boolean, host: string, open: boolean, out: string, port: number, server: string, version: boolean }}
 */
export function parseCliArgs(argv) {
	const { positionals, values } = parseArgs({
		allowNegative: true,
		allowPositionals: true,
		args: argv,
		options: {
			force: { type: 'boolean' },
			help: { short: 'h', type: 'boolean' },
			host: { type: 'string' },
			open: { type: 'boolean' },
			out: { type: 'string' },
			port: { type: 'string' },
			server: { type: 'string' },
			version: { short: 'v', type: 'boolean' }
		},
		strict: true
	});
	const [verb, ...rest] = positionals;

	if ((verb !== undefined && verb !== 'start' && verb !== 'build') || rest.length > 0) {
		throw new Error(`unknown command: ${positionals.join(' ')}`);
	}

	const command = verb === 'build' ? 'build' : 'serve';
	const misplaced = (command === 'build' ? SERVE_OPTIONS : BUILD_OPTIONS).find(
		(name) => name in values
	);

	if (misplaced) {
		throw new Error(
			`--${misplaced} does not apply to ${command === 'build' ? 'build' : 'serving'}`
		);
	}

	const port = Number(values.port ?? DEFAULTS.port);

	if (!Number.isInteger(port) || port < 0 || port > 65535) {
		throw new Error(`invalid --port: ${values.port}`);
	}

	let server = '';

	if (values.server !== undefined) {
		let parsed;

		try {
			parsed = new URL(values.server);
		} catch {
			parsed = null;
		}

		if (!parsed || (parsed.protocol !== 'http:' && parsed.protocol !== 'https:')) {
			throw new Error(`invalid --server, expected an http(s) URL: ${values.server}`);
		}

		server = `${parsed.origin}${parsed.pathname.replace(/\/+$/, '')}`;
	}

	return {
		command,
		force: values.force ?? false,
		help: values.help ?? false,
		host: values.host ?? DEFAULTS.host,
		open: values.open ?? true,
		out: values.out ?? DEFAULTS.out,
		port,
		server,
		version: values.version ?? false
	};
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
 * @param {http.ServerResponse} res
 * @param {number} status
 * @param {string} message
 */
function sendError(res, status, message) {
	res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8' });
	res.end(`${message}\n`);
}

/**
 * The app's config.json for a llama-server at `server`.
 * @param {string} server
 */
function configJson(server) {
	return `${JSON.stringify({ server })}\n`;
}

/**
 * The llama-server on its default port, on the host a request was made to: localhost when
 * opened here, the same LAN or Tailscale address when opened from another device.
 * @param {http.IncomingMessage} req
 */
function defaultServer(req) {
	const { hostname } = new URL(`http://${req.headers.host ?? 'localhost'}`);

	return `http://${hostname}:${DEFAULTS.serverPort}`;
}

/**
 * The HTTP server: the files of `dist`, read-only. It answers config.json itself, so the app
 * talks to `server`, or by default to llama-server's default port, instead of this address.
 * @param {{ distDir?: string, server?: string }} [options]
 */
export function createChatServer({ distDir = DIST_DIR, server = '' } = {}) {
	const root = normalize(distDir);

	return http.createServer(async (req, res) => {
		const { pathname } = new URL(req.url ?? '/', 'http://localhost');

		if (req.method !== 'GET' && req.method !== 'HEAD') {
			res.setHeader('Allow', 'GET, HEAD');
			sendError(res, 405, 'Method Not Allowed');

			return;
		}

		if (pathname === CONFIG_PATH) {
			const body = configJson(server || defaultServer(req));

			res.writeHead(200, {
				'Cache-Control': CACHE_REVALIDATE,
				'Content-Length': Buffer.byteLength(body),
				'Content-Type': MIME_TYPES['.json']
			});
			res.end(req.method === 'HEAD' ? undefined : body);

			return;
		}

		const file = resolveStatic(root, pathname);
		const stats = file ? await stat(file).catch(() => null) : null;

		if (file && stats?.isFile()) {
			sendFile(req, res, file, pathname, stats);

			return;
		}

		sendError(res, 404, 'Not Found');
	});
}

/**
 * Writes the app to a folder: the files of `dist` without source maps, ready for any static
 * file server. Refuses a folder that is not empty unless `force` is set, and a folder in `dist`.
 * With `server`, its config.json names that llama-server.
 * @param {{ distDir?: string, force?: boolean, out: string, server?: string }} options
 * @returns {Promise<string>} the absolute path of the folder
 */
export async function buildStatic({ distDir = DIST_DIR, force = false, out, server = '' }) {
	const source = resolve(distDir);
	const target = resolve(out);
	const inside = relative(source, target);

	if (inside === '' || !(inside.startsWith('..') || isAbsolute(inside))) {
		throw new Error(`cannot build into ${target}, it is inside the app's own files`);
	}

	const entries = await readdir(target).catch(() => []);

	if (entries.length > 0 && !force) {
		throw new Error(`${target} is not empty, pick another --out or add --force`);
	}

	await cp(source, target, { filter: (file) => !file.endsWith('.map'), recursive: true });

	if (server) await writeFile(join(target, CONFIG_PATH), configJson(server));

	return target;
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

	if (options.command === 'build') {
		try {
			const target = await buildStatic({
				force: options.force,
				out: options.out,
				server: options.server
			});
			const shown = relative(process.cwd(), target) || '.';

			console.log(`Llama Chat ${packageVersion()} written to ${target}`);
			console.log(`Serve it with any static file server, e.g.: npx http-server ${shown}`);
		} catch (error) {
			console.error(`llama-chat: ${error instanceof Error ? error.message : error}`);
			process.exitCode = 1;
		}

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

			console.log(
				`Talking to llama-server on ${options.server || `port ${DEFAULTS.serverPort}, pick another with --server`}`
			);

			console.log(
				`If llama-server restricts origins (tools or MCP on), start it with --cors-origins ${url.slice(0, -1)}`
			);

			if (options.open) openBrowser(url);

			resolve(server);
		});
	});
}
