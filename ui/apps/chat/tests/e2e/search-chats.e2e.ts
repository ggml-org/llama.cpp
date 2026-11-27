import { mockLlamaServer } from '../../demo/fixtures/llama-server.ts';
import { generateTrace } from '../../demo/fixtures/traces.ts';
import { importTraces } from '../../demo/helpers.ts';
import { expect, test } from '@playwright/test';
import { writeFileSync } from 'node:fs';

test('search finds a chat by its title and opens it', async ({ page }, testInfo) => {
	const file = testInfo.outputPath('traces.jsonl');

	writeFileSync(file, [generateTrace('long-chat'), generateTrace('agentic-debugging')].join('\n'));
	await mockLlamaServer(page);
	await page.goto('/');
	await importTraces(page, file);
	await page.reload();

	await page.getByLabel('Expand navigation').click();
	await page.getByRole('button', { name: /^Search/ }).click();

	const search = page.getByRole('dialog', { name: 'Search' });

	await search.getByRole('combobox', { name: 'Search' }).fill('flaky');
	await search.getByRole('option', { name: /Fix the flaky cache tests/ }).click();

	await expect(page.getByRole('main')).toContainText('All 48 cache tests pass now');
});
