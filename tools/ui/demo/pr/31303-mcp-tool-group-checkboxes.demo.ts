// https://github.com/ggml-org/llama.cpp/pull/31303
import { test } from '../helpers.ts';

test('demo', async ({ demo, page }) => {
	await demo.click(page.getByRole('button', { exact: true, name: 'Settings' }));

	const dialog = page.getByRole('dialog');

	await demo.click(dialog.getByText('Tools', { exact: true }).filter({ visible: true }));
	await demo.caption('MCP servers now get a group checkbox in Tools settings');

	const weather = dialog.getByRole('button', { name: /Weather/ });

	await demo.click(weather);
	await demo.click(weather.getByRole('checkbox'), 1000);
	await demo.caption('Turning a server off disables all of its tools');

	await demo.press('Escape');
	await demo.click(page.getByRole('button', { name: 'Add files, prompts, tools or MCP Servers' }));
	await demo.hover(page.getByRole('menuitem', { name: 'Tools' }));
	await demo.caption('The chat tools menu uses the same setting');

	const menu = page.getByRole('menu').last();

	await demo.click(menu.getByRole('button', { name: /^Notes/ }));
	await demo.click(menu.getByRole('button', { name: 'create_note' }), 1000);
	await demo.caption('Partly enabled groups keep the checked colors');
	await demo.pause(1000);
});
