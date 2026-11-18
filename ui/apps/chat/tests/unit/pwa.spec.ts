import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const DIST_DIR = resolve(__dirname, '../../dist');
const distExists = existsSync(DIST_DIR);

// PWA Build Output tests are integration tests that require a built dist/.
// CI builds first then runs these tests; local devs should run `npm run build` or use `npm run test:pwa`.
describe('PWA Build Output', () => {
	if (!distExists) {
		console.warn(`⚠ Skipping PWA Build Output tests - dist/ not found (run 'npm run build' first)`);
		it('skipped - dist/ not found', () => {});

		return;
	}

	const swContent = readFileSync(resolve(DIST_DIR, 'service-worker.js'), 'utf-8');
	const legacySwContent = readFileSync(resolve(DIST_DIR, 'sw.js'), 'utf-8');
	const indexContent = readFileSync(resolve(DIST_DIR, 'index.html'), 'utf-8');

	describe('Core files exist', () => {
		it('service worker (service-worker.js) exists', () => {
			expect(
				existsSync(resolve(DIST_DIR, 'service-worker.js')),
				'service-worker.js not found'
			).toBeTruthy();
		});

		it('legacy service worker (sw.js) exists', () => {
			expect(existsSync(resolve(DIST_DIR, 'sw.js')), 'sw.js not found').toBeTruthy();
		});

		it('manifest.webmanifest exists', () => {
			expect(
				existsSync(resolve(DIST_DIR, 'manifest.webmanifest')),
				'manifest.webmanifest not found'
			).toBeTruthy();
		});

		it('SvelteKit bundle.js exists in _app/immutable/', () => {
			// SvelteKit generates hashed bundle names in _app/immutable/
			const appDir = resolve(DIST_DIR, '_app', 'immutable');

			expect(existsSync(appDir), '_app/immutable/ not found').toBeTruthy();
			const files = readdirSync(appDir).filter((f) => f.startsWith('bundle.') && f.endsWith('.js'));

			expect(files.length).toBeGreaterThan(0);
		});

		it('SvelteKit bundle.css exists in _app/immutable/assets/', () => {
			// SvelteKit generates hashed CSS bundles in _app/immutable/assets/
			const cssDir = resolve(DIST_DIR, '_app', 'immutable', 'assets');

			expect(existsSync(cssDir), '_app/immutable/assets/ not found').toBeTruthy();
			const files = readdirSync(cssDir).filter(
				(f) => f.startsWith('bundle.') && f.endsWith('.css')
			);

			expect(files.length).toBeGreaterThan(0);
		});

		it('version.json exists in _app/', () => {
			// SvelteKit stores version.json in _app directory
			expect(
				existsSync(resolve(DIST_DIR, '_app', 'version.json')),
				'_app/version.json not found'
			).toBeTruthy();
		});
	});

	describe('version.json content', () => {
		it('has valid JSON with version field', () => {
			const content = readFileSync(resolve(DIST_DIR, '_app', 'version.json'), 'utf-8');
			const parsed = JSON.parse(content);

			expect(parsed).toHaveProperty('version');
			expect(typeof parsed.version).toBe('string');
			expect(parsed.version.length).toBeGreaterThan(0);
		});
	});

	describe('Service worker content', () => {
		it('precache contains SvelteKit bundle.js with content hash', () => {
			expect(swContent).toMatch(/["'`]_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js["'`]/);
		});

		it('precache contains SvelteKit bundle.css with content hash', () => {
			expect(swContent).toMatch(/["'`]_app\/immutable\/assets\/bundle\.[a-zA-Z0-9_-]+\.css["'`]/);
		});

		it('precache contains the app shell, _app/version.json and manifest.webmanifest', () => {
			expect(swContent).toMatch(/["'`]\.\/["'`]/);
			expect(swContent).toMatch(/["'`]_app\/version\.json["'`]/);
			expect(swContent).toMatch(/["'`]manifest\.webmanifest["'`]/);
		});

		it('activates a waiting worker on SKIP_WAITING', () => {
			expect(swContent).toMatch(/SKIP_WAITING/);
			expect(swContent).toMatch(/skipWaiting\(\)/);
		});

		it('precaches from the network, not the HTTP cache', () => {
			// a stale index.html from the HTTP cache would point at the previous build's bundle
			expect(swContent).toMatch(/cache:\s*["'`]reload["'`]/);
		});

		it('does not route API requests', () => {
			// only precached files are served from cache, so API calls always hit the server
			expect(swContent).not.toMatch(/v1\//);
		});
	});

	describe('Legacy service worker content', () => {
		it('unregisters itself and clears caches', () => {
			expect(legacySwContent).toMatch(/registration\.unregister\(\)/);
			expect(legacySwContent).toMatch(/caches\.delete/);
		});
	});

	describe('index.html content', () => {
		it('has modulepreload link for SvelteKit bundle with content hash', () => {
			expect(indexContent).toBeTruthy();
			// SvelteKit generates hashed bundle names in _app/immutable/
			expect(indexContent).toMatch(/href="(\.\/|\/)_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js"/);
		});

		it('has stylesheet link for SvelteKit bundle.css with content hash', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(
				/href="(\.\/|\/)_app\/immutable\/assets\/bundle\.[a-zA-Z0-9_-]+\.css"/
			);
		});

		it('has dynamic import for SvelteKit bundle with content hash', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(
				/import\("(\.\/|\/)_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js"\)/
			);
		});

		it('has __sveltekit__ variable (SvelteKit adds hash suffix)', () => {
			expect(indexContent).toBeTruthy();
			// SvelteKit 2.x uses __sveltekit__ as base with random suffix
			expect(indexContent).toMatch(/__sveltekit_[a-zA-Z0-9-]+/);
		});

		it('has PWA manifest link', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/rel="manifest" href="(\.?\/)?manifest\.webmanifest"/);
		});

		it('has apple-touch-icon link', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/rel="apple-touch-icon"/);
		});

		it('has _app paths for SvelteKit bundles', () => {
			expect(indexContent).toBeTruthy();
			// SvelteKit uses _app paths for hashed assets
			expect(indexContent).toMatch(/_app\//);
		});
	});

	describe('SvelteKit _app directory', () => {
		it('_app directory exists (SvelteKit uses it for hashed assets)', () => {
			expect(existsSync(resolve(DIST_DIR, '_app'))).toBeTruthy();
		});
	});

	describe('Static assets', () => {
		it('has favicon.ico', () => {
			expect(existsSync(resolve(DIST_DIR, 'favicon.ico'))).toBeTruthy();
		});

		it('has PWA icons', () => {
			expect(existsSync(resolve(DIST_DIR, 'pwa-64x64.png'))).toBeTruthy();
			expect(existsSync(resolve(DIST_DIR, 'pwa-192x192.png'))).toBeTruthy();
			expect(existsSync(resolve(DIST_DIR, 'pwa-512x512.png'))).toBeTruthy();
		});
	});
});
