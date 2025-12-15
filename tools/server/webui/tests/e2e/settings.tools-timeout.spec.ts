import { test, expect } from '@playwright/test';

test('settings shows code interpreter timeout field (disabled until tool enabled)', async ({
	page
}) => {
	// Ensure config is present before app boot
	await page.addInitScript(() => {
		localStorage.setItem(
			'LlamaCppWebui.config',
			JSON.stringify({
				enableCalculatorTool: true,
				enableCodeInterpreterTool: false
			})
		);
	});

	// Mock /props and model list so UI initializes in static preview
	await page.route('**/props**', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				role: 'model',
				system_prompt: null,
				default_generation_settings: { params: {}, n_ctx: 4096 }
			})
		});
	});
	await page.route('**/v1/models**', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({ object: 'list', data: [{ id: 'mock-model', object: 'model' }] })
		});
	});

	await page.goto('http://localhost:8181/');

	// Open settings dialog (header has only the settings button)
	await page.locator('header button').first().click();
	await expect(page.getByRole('dialog')).toBeVisible({ timeout: 5000 });

	// Navigate to Tools section
	await page.getByRole('button', { name: 'Tools' }).first().click();
	await expect(page.getByRole('heading', { name: 'Tools' })).toBeVisible();

	// Tool toggle exists
	await expect(page.getByText('Code Interpreter (JavaScript)')).toBeVisible();

	// Tool-defined setting always visible
	await expect(page.getByText('Code interpreter timeout (seconds)')).toBeVisible();

	const timeoutInput = page.locator('#codeInterpreterTimeoutSeconds');
	await expect(timeoutInput).toBeVisible();
	await expect(timeoutInput).toBeDisabled();

	// Enable tool
	await page.locator('label[for="enableCodeInterpreterTool"]').click();
	await expect(page.locator('#enableCodeInterpreterTool')).toHaveAttribute('data-state', 'checked');
	await expect(timeoutInput).toBeEnabled();

	// Default value
	await expect(timeoutInput).toHaveValue('30');
});
