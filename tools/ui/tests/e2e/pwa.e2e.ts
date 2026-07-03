import { expect, test } from '@playwright/test';

test.describe('PWA Service Worker', () => {
	test('service worker is registered', async ({ page }) => {
		await page.goto('/');

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

		await page.evaluate(async () => {
			await navigator.serviceWorker.ready;
		});

		const swActive = await page.evaluate(async () => {
			const reg = await navigator.serviceWorker.ready;
			return reg.active?.scriptURL ?? null;
		});

		expect(swActive).toBeTruthy();

		const swResponse = await page.request.get(swActive!);
		const swContent = await swResponse.text();

		// Precache contains SvelteKit content-hashed bundle paths
		expect(swContent).toMatch(/"_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js"/);
		expect(swContent).toMatch(/"_app\/immutable\/assets\/bundle\.[a-zA-Z0-9_-]+\.css"/);
		expect(swContent).toMatch(/"manifest\.webmanifest"/);
		expect(swContent).toMatch(/"_app\/version\.json"/);
		// NavigationRoute is intentionally absent — server API endpoints
		// (e.g. /slots, /models) must not be intercepted by the PWA and
		// should return JSON directly from the server.
		expect(swContent).not.toMatch(/NavigationRoute/);
		expect(swContent).toMatch(/api-cache/);
	});

	test('offline mode - page loads when offline after caching', async ({ browser }) => {
		const context = await browser.newContext();
		const offlinePage = await context.newPage();

		await offlinePage.goto('/');
		await offlinePage.waitForLoadState('networkidle');

		await offlinePage.evaluate(async () => {
			await navigator.serviceWorker.ready;
		});

		await offlinePage.waitForTimeout(2000);

		await context.setOffline(true);
		await offlinePage.goto('/');

		const bodyText = await offlinePage.locator('body').textContent();
		expect(bodyText).toBeTruthy();

		await context.close();
	});

	test('version.json is accessible and contains version', async ({ page }) => {
		const versionResponse = await page.request.get('/_app/version.json');
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
		expect(manifest).toHaveProperty('start_url', './');
		expect(manifest).toHaveProperty('display', 'standalone');
		expect(manifest.icons).toBeTruthy();
		expect(manifest.icons.length).toBeGreaterThan(0);
	});

	test('index.html contains content-hashed bundle references', async ({ page }) => {
		const response = await page.request.get('/');
		expect(response.ok()).toBeTruthy();

		const html = await response.text();

		// SvelteKit outputs content-hashed bundle names in _app/immutable/
		expect(html).toMatch(/href="(\.\/|\/)_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js"/);
		expect(html).toMatch(/href="(\.\/|\/)_app\/immutable\/assets\/bundle\.[a-zA-Z0-9_-]+\.css"/);
		expect(html).toMatch(/import\("(\.\/|\/)_app\/immutable\/bundle\.[a-zA-Z0-9_-]+\.js"\)/);
	});
});
