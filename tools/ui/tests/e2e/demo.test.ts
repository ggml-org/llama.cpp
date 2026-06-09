import { expect, test } from '@playwright/test';

test('home page loads correctly', async ({ page }) => {
	await page.goto('/');
	// Wait for the greeting to become visible (stores need time to initialize)
	await expect(page.locator('h1', { hasText: /Hello there/ })).toBeVisible();
});
