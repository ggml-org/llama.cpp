import { expect, test } from '@playwright/test';

test.describe('PWA Service Worker', () => {
	test('service worker is registered', async ({ page }) => {
		await page.goto('/');

		// Wait for service worker to be ready
		const swURL = await page.evaluate(async () => {
			const registration = await Promise.race([
				// eslint-disable-next-line @typescript-eslint/ban-ts-comment
				// @ts-ignore - type inference differs from browser runtime
				navigator.serviceWorker.ready,
				new Promise((_, reject) =>
					setTimeout(() => reject(new Error('Service worker registration failed: timeout')), 15000)
				)
			]);
			// @ts-expect-error registration is of type unknown
			return registration.active?.scriptURL;
		});

		expect(swURL).toBeTruthy();
		expect(swURL).toContain('/sw.js');
	});

	test('service worker has precache configured', async ({ page }) => {
		await page.goto('/');

		// Wait for SW to be active
		await page.evaluate(async () => {
			await navigator.serviceWorker.ready;
		});

		// Get the SW registration and check its script URL
		const swActive = await page.evaluate(async () => {
			const reg = await navigator.serviceWorker.ready;
			return reg.active?.scriptURL ?? null;
		});

		expect(swActive).toBeTruthy();

		// Fetch the SW content directly
		const swResponse = await page.request.get(swActive!);
		const swContent = await swResponse.text();

		// Verify precache contains versioned bundle URLs
		expect(swContent).toMatch(/"\.\/bundle\.js\?v=/);
		expect(swContent).toMatch(/"\.\/bundle\.css\?v=/);

		// Verify other expected precache entries
		expect(swContent).toMatch(/"manifest\.webmanifest"/);
		expect(swContent).toMatch(/"version\.json"/);

		// Verify SW has navigation route and runtime caching
		expect(swContent).toMatch(/NavigationRoute/);
		expect(swContent).toMatch(/api-cache/);
	});

	test('offline mode - page loads when offline after caching', async ({ browser }) => {
		const context = await browser.newContext();
		const offlinePage = await context.newPage();

		// First, load the page to ensure SW registers
		await offlinePage.goto('/');
		await offlinePage.waitForLoadState('networkidle');

		// Wait for SW to be ready
		await offlinePage.evaluate(async () => {
			await navigator.serviceWorker.ready;
		});

		// Wait for cache to potentially populate
		await offlinePage.waitForTimeout(2000);

		// Enable offline mode
		await context.setOffline(true);

		// Try to navigate - with hash routing, the app should still be accessible
		// since the SW has precached the entry point
		await offlinePage.goto('/');

		// Page should have some content (not blank)
		const bodyText = await offlinePage.locator('body').textContent();
		// The page should at least have something rendered
		expect(bodyText).toBeTruthy();

		await context.close();
	});

	test('version.json is accessible and contains version', async ({ page }) => {
		// Fetch version.json directly
		const versionResponse = await page.request.get('/version.json');
		expect(versionResponse.ok()).toBeTruthy();

		const versionData = await versionResponse.json();
		expect(versionData).toHaveProperty('version');
		expect(typeof versionData.version).toBe('string');
		expect(versionData.version.length).toBeGreaterThan(0);
	});

	test('manifest.webmanifest is accessible and valid', async ({ page }) => {
		const response = await page.request.get('/manifest.webmanifest');
		expect(response.ok()).toBeTruthy();

		const manifest = await response.json();
		expect(manifest).toHaveProperty('name', 'llama-ui');
		expect(manifest).toHaveProperty('short_name', 'llama-ui');
		expect(manifest).toHaveProperty('start_url', './?pwa=1');
		expect(manifest).toHaveProperty('display', 'standalone');
		expect(manifest.icons).toBeTruthy();
		expect(manifest.icons.length).toBeGreaterThan(0);
	});

	test('bundle files are accessible with version query', async ({ page }) => {
		// Get version from version.json
		const versionResponse = await page.request.get('/version.json');
		const { version } = await versionResponse.json();

		// Try to fetch bundle.js with version param
		const bundleResponse = await page.request.get(`/bundle.js?v=${version}`);
		expect(bundleResponse.ok()).toBeTruthy();

		// Try to fetch bundle.css with version param
		const cssResponse = await page.request.get(`/bundle.css?v=${version}`);
		expect(cssResponse.ok()).toBeTruthy();
	});

	test('index.html contains versioned bundle references', async ({ page }) => {
		const response = await page.request.get('/index.html');
		expect(response.ok()).toBeTruthy();

		const html = await response.text();

		// Should have modulepreload with version or content-hashed bundle.js (relative path)
		expect(html).toMatch(
			/href="\.\/bundle\.js\?v=[a-zA-Z0-9._-]+|href="\.\/bundle\.js\?[a-zA-Z0-9_-]+/
		);
		// Should have stylesheet with version or content-hashed bundle.css (relative path)
		expect(html).toMatch(
			/href="\.\/bundle\.css\?v=[a-zA-Z0-9._-]+|href="\.\/bundle\.css\?[a-zA-Z0-9_-]+/
		);
		// Should have dynamic import with version or content hash
		expect(html).toMatch(
			/import\("\.\/bundle\.js\?v=[a-zA-Z0-9._-]+|import\("\.\/bundle\.js\?[a-zA-Z0-9_-]+/
		);
	});
});
