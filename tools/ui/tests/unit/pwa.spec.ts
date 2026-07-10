import { existsSync, readFileSync, readdirSync } from 'node:fs';
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

	const swContent = readFileSync(resolve(DIST_DIR, 'sw.js'), 'utf-8');
	const indexContent = readFileSync(resolve(DIST_DIR, 'index.html'), 'utf-8');

	describe('Core files exist', () => {
		it('service worker (sw.js) exists', () => {
			expect(existsSync(resolve(DIST_DIR, 'sw.js')), 'sw.js not found').toBeTruthy();
		});

		it('workbox library exists (hashed filename)', () => {
			// SvelteKit generates workbox-{hash}.js files
			const files = readdirSync(DIST_DIR).filter((f) => f.match(/^workbox-[^.]+\.js$/));
			expect(files.length).toBeGreaterThan(0);
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
		it('service worker has minified self.define format', () => {
			expect(swContent).toBeTruthy();
			// SvelteKit's workbox-plugin-sveltekit produces a minified SW with self.define
			expect(swContent).toMatch(/if\(!self.define\)/);
		});

		it('references hashed workbox file (SvelteKit build output)', () => {
			expect(swContent).toBeTruthy();
			// SvelteKit's workbox-plugin-sveltekit references hashed workbox files
			expect(swContent).toMatch(/define\(\["\.\/workbox-[a-zA-Z0-9]+"\]/);
		});

		it('precache contains SvelteKit bundle.js with content hash', () => {
			expect(swContent).toBeTruthy();
			// SvelteKit uses content-hashed bundle names in _app/immutable/
			expect(swContent).toMatch(/"_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js"/);
		});

		it('precache contains SvelteKit bundle.css with content hash', () => {
			expect(swContent).toBeTruthy();
			// SvelteKit uses content-hashed CSS bundle names in _app/immutable/assets/
			expect(swContent).toMatch(/"_app\/immutable\/assets\/bundle\.[a-zA-Z0-9_-]+\.css"/);
		});

		it('precache contains _app/version.json', () => {
			expect(swContent).toBeTruthy();
			// SvelteKit stores version.json in _app directory
			expect(swContent).toMatch(/"_app\/version\.json"/);
		});

		it('precache contains manifest.webmanifest', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/"manifest\.webmanifest"/);
		});

		it('no navigation route — API endpoints bypass PWA', () => {
			expect(swContent).toBeTruthy();
			// NavigationRoute is intentionally absent so direct browser
			// navigation to server API endpoints returns JSON, not HTML.
			expect(swContent).not.toMatch(/NavigationRoute/);
		});

		it('has runtime caching for API routes', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/api-cache/);
			expect(swContent).toMatch(/NetworkFirst/);
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

	describe('Hashed workbox files', () => {
		it('workbox-*.js files exist in dist root (SvelteKit build output)', () => {
			const files = readdirSync(DIST_DIR).filter((f) => f.match(/^workbox-[^.]+\.js$/));
			expect(files.length).toBeGreaterThan(0);
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
