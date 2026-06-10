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

		it('workbox library exists', () => {
			expect(existsSync(resolve(DIST_DIR, 'workbox.js')), 'workbox.js not found').toBeTruthy();
		});

		it('manifest.webmanifest exists', () => {
			expect(
				existsSync(resolve(DIST_DIR, 'manifest.webmanifest')),
				'manifest.webmanifest not found'
			).toBeTruthy();
		});

		it('bundle.js exists', () => {
			expect(existsSync(resolve(DIST_DIR, 'bundle.js')), 'bundle.js not found').toBeTruthy();
		});

		it('bundle.css exists', () => {
			expect(existsSync(resolve(DIST_DIR, 'bundle.css')), 'bundle.css not found').toBeTruthy();
		});

		it('version.json exists', () => {
			expect(existsSync(resolve(DIST_DIR, 'version.json')), 'version.json not found').toBeTruthy();
		});
	});

	describe('version.json content', () => {
		it('has valid JSON with version field', () => {
			const content = readFileSync(resolve(DIST_DIR, 'version.json'), 'utf-8');
			const parsed = JSON.parse(content);
			expect(parsed).toHaveProperty('version');
			expect(typeof parsed.version).toBe('string');
			expect(parsed.version.length).toBeGreaterThan(0);
		});
	});

	describe('Service worker content', () => {
		it('has build version comment', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/^\/\/ Build: /);
		});

		it('references workbox.js (not workbox-*.js)', () => {
			expect(swContent).toBeTruthy();
			// Should NOT have any hashed workbox references
			expect(swContent).not.toMatch(/"\.\/workbox-[a-z0-9]+"/);
			// Should have the renamed workbox.js (uses double quotes in minified SW)
			// eslint-disable-next-line no-useless-escape
			expect(swContent).toMatch(/define\(\[\"\.\/workbox\"\]/);
		});

		it('precache contains bundle.js with cache param', () => {
			expect(swContent).toBeTruthy();
			// Should have precache entry with ?cache=true for SW caching
			expect(swContent).toMatch(/"\.\/bundle\.js\?cache=true"/);
		});

		it('precache contains bundle.css with cache param', () => {
			expect(swContent).toBeTruthy();
			// Should have precache entry with ?cache=true for SW caching
			expect(swContent).toMatch(/"\.\/bundle\.css\?cache=true"/);
		});

		it('precache contains version.json (not _app/version.json)', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/"version\.json"/);
			expect(swContent).not.toMatch(/"_app\/version\.json"/);
		});

		it('precache contains manifest.webmanifest', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/"manifest\.webmanifest"/);
		});

		it('has navigation route registered', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/NavigationRoute/);
		});

		it('has runtime caching for API routes', () => {
			expect(swContent).toBeTruthy();
			expect(swContent).toMatch(/api-cache/);
			expect(swContent).toMatch(/NetworkFirst/);
		});
	});

	describe('index.html content', () => {
		it('has modulepreload link for bundle.js with ?cache=true', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/href="\.\/bundle\.js\?cache=true"/);
		});

		it('has stylesheet link for bundle.css with ?cache=true', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/href="\.\/bundle\.css\?cache=true"/);
		});

		it('has dynamic import for bundle.js with ?cache=true', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/import\("\.\/bundle\.js\?cache=true"\)/);
		});

		it('has __sveltekit__ (not __sveltekit_<hash>)', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/__sveltekit__/);
			expect(indexContent).not.toMatch(/__sveltekit_[a-z0-9]+/);
		});

		it('has PWA manifest link', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/rel="manifest" href="(\.?\/)?manifest\.webmanifest"/);
		});

		it('has apple-touch-icon link', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).toMatch(/rel="apple-touch-icon"/);
		});

		it('does not have _app paths', () => {
			expect(indexContent).toBeTruthy();
			expect(indexContent).not.toMatch(/\/_app\//);
		});
	});

	describe('No _app directory', () => {
		it('_app directory should not exist', () => {
			expect(existsSync(resolve(DIST_DIR, '_app'))).toBeFalsy();
		});
	});

	describe('No hashed workbox files', () => {
		it('no workbox-*.js files in dist root', () => {
			const files = readdirSync(DIST_DIR).filter((f) => f.match(/^workbox-[^.]+\.js$/));
			expect(files).toHaveLength(0);
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

		it('has loading.html fallback page', () => {
			expect(existsSync(resolve(DIST_DIR, 'loading.html'))).toBeTruthy();
		});
	});
});
