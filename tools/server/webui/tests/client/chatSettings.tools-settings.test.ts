import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { SETTING_CONFIG_DEFAULT } from '$lib/constants/settings-config';
import ChatSettings from '$lib/components/app/chat/ChatSettings/ChatSettings.svelte';
import { tick } from 'svelte';

async function selectToolsSection(container: HTMLElement) {
	const toolsButtons = Array.from(container.querySelectorAll('button')).filter((b) =>
		(b.textContent ?? '').includes('Tools')
	);
	for (const b of toolsButtons) b.click();
	await tick();
}

describe('ChatSettings tool-defined settings', () => {
	it('shows a tool setting when the tool is enabled', async () => {
		settingsStore.config = { ...SETTING_CONFIG_DEFAULT, enableCodeInterpreterTool: true };

		const { container } = render(ChatSettings, { target: document.body, props: {} });
		await selectToolsSection(container);

		expect(container.textContent).toContain('Code Interpreter (JavaScript)');
		expect(container.textContent).toContain('Code interpreter timeout (seconds)');
	});

	it('shows the tool setting disabled when tool is off, then enables it', async () => {
		settingsStore.config = { ...SETTING_CONFIG_DEFAULT, enableCodeInterpreterTool: false };

		const { container } = render(ChatSettings, { target: document.body, props: {} });
		await selectToolsSection(container);

		expect(container.textContent).toContain('Code Interpreter (JavaScript)');
		expect(container.textContent).toContain('Code interpreter timeout (seconds)');
		const timeoutInput = container.querySelector(
			'input#codeInterpreterTimeoutSeconds'
		) as HTMLInputElement | null;
		expect(timeoutInput).toBeTruthy();
		expect(timeoutInput?.disabled).toBe(true);

		const enableLabel = container.querySelector(
			'label[for="enableCodeInterpreterTool"]'
		) as HTMLElement | null;
		enableLabel?.click();
		await tick();

		expect(container.textContent).toContain('Code interpreter timeout (seconds)');
		expect(timeoutInput?.disabled).toBe(false);
	});
});
