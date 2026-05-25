import { expect, test } from '@playwright/test';

test('home page loads', async ({ page }) => {
	await page.goto('/');
	// Wait for the SPA to hydrate — the chat form is rendered by the hydrated app
	await expect(page.locator('form').first()).toBeVisible();
});
