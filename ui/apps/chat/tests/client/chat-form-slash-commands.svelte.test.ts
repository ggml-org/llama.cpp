// Guards the slash-command dispatch contract: commands dispatch only on
// explicit selection (Enter/click in the picker), never mid-typing.
// Typing `/model is broken` is prose until the command is picked - the
// buffer must survive; only an actual selection consumes the token.

import ChatFormPickersHarness from './components/ChatFormPickersHarness.svelte';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

describe('slash command dispatch', () => {
	it('does not dispatch or clear the buffer when a space follows the name', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/model is broken');
		await tick();

		const pickers = screen.component.getPickers();

		expect(screen.component.getValue()).toBe('/model is broken');
		expect(screen.component.getCalls()).not.toContain('openModelSelector');
		expect(screen.component.getCalls().some((c) => c.startsWith('setValue:'))).toBe(false);
		expect(pickers.isCommandPickerOpen).toBe(true);
		expect(pickers.commandQuery).toBe('model');
	});

	it('dispatches /model on explicit selection and consumes the token', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/model is broken');
		await tick();

		const pickers = screen.component.getPickers();
		const model = pickers.availableCommands.find((c) => c.name === 'model');

		if (!model) throw new Error('model command missing');

		pickers.handleCommandSelect(model);
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(screen.component.getCalls()).toContain('openModelSelector');
		expect(pickers.isCommandPickerOpen).toBe(false);
	});

	it('seeds the prompt picker search from the token args on selection', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/prompt weather');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isPromptPickerOpen).toBe(false);

		const prompt = pickers.availableCommands.find((c) => c.name === 'prompt');

		if (!prompt) throw new Error('prompt command missing');

		pickers.handleCommandSelect(prompt);
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(pickers.isPromptPickerOpen).toBe(true);
		expect(pickers.promptSearchQuery).toBe('weather');
	});

	it('normalizes a partial /cwd token on selection and keeps it in the buffer', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/cw docs');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isWorkingDirectoryPickerOpen).toBe(false);

		const cwd = pickers.availableCommands.find((c) => c.name === 'cwd');

		if (!cwd) throw new Error('cwd command missing');

		pickers.handleCommandSelect(cwd);
		await tick();

		expect(pickers.isWorkingDirectoryPickerOpen).toBe(true);
		expect(pickers.workingDirectoryQuery).toBe('docs');
		expect(screen.component.getValue()).toBe('/cwd docs');
	});

	it('syncs the /cwd token into the picker search while the picker is open', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		const pickers = screen.component.getPickers();
		const cwd = pickers.availableCommands.find((c) => c.name === 'cwd');

		if (!cwd) throw new Error('cwd command missing');

		screen.component.type('/cwd docs');
		pickers.handleCommandSelect(cwd);
		await tick();

		screen.component.type('/cwd docs/sub');
		await tick();

		expect(pickers.isWorkingDirectoryPickerOpen).toBe(true);
		expect(pickers.workingDirectoryQuery).toBe('docs/sub');
		expect(pickers.isCommandPickerOpen).toBe(false);
	});

	it('abandons the /cwd picker when the token is edited away from /cwd', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		const pickers = screen.component.getPickers();
		const cwd = pickers.availableCommands.find((c) => c.name === 'cwd');

		if (!cwd) throw new Error('cwd command missing');

		screen.component.type('/cwd docs');
		pickers.handleCommandSelect(cwd);
		await tick();

		screen.component.type('/cwdd docs');
		await tick();

		expect(pickers.isWorkingDirectoryPickerOpen).toBe(false);
	});
});
