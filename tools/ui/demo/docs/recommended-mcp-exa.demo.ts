// Needs network access and a model that calls tools.
import { test } from '../helpers.ts';

test.use({ settings: { mcpServers: '[]' } });

test('demo', async ({ demo, page }) => {
	await demo.click(page.getByRole('button', { exact: true, name: 'MCP' }));
	await demo.click(page.getByRole('button', { name: 'Add New Server' }));
	await demo.click(page.getByText('Exa Search the web and fetch'));
	await demo.click(page.getByRole('button', { name: 'Save' }), 1500);
	await demo.click(page.getByRole('button', { name: 'New chat' }));
	await demo.type(
		page.getByRole('textbox', { name: 'Type a message...' }),
		'Who won last FIFA World Cup?'
	);
	await demo.click(page.getByRole('button', { name: 'Send' }));
	await demo.click(page.getByRole('button', { name: 'Allow once' }).first());
	await page.getByRole('button', { name: 'Send' }).waitFor({ timeout: 60000 });
	await demo.pause(2000);
});
