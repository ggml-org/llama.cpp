// Guards the slash-command dispatch contract: commands dispatch only on
// explicit selection (Enter/click in the picker), never mid-typing.
// Typing `/model is broken` is prose: `/model` takes no argument, so the
// words after it close the picker and the buffer survives; only an actual
// selection consumes the token.

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
		expect(pickers.isCommandPickerOpen).toBe(false);
	});

	it('dispatches /model on explicit selection and consumes the token', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/model');
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

	it('opens the system message editor on /system-message and consumes the token', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/system');
		await tick();

		const pickers = screen.component.getPickers();
		const command = pickers.availableCommands.find((c) => c.name === 'system-message');

		if (!command) throw new Error('system-message command missing');

		pickers.handleCommandSelect(command);
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(screen.component.getCalls()).toContain('onSystemMessage');
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

describe('commands anywhere in the message', () => {
	it('opens the command picker for a command typed after other words', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.typeAt('fix the build /mod|');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isCommandPickerOpen).toBe(true);
		expect(pickers.commandQuery).toBe('mod');
	});

	it('takes only the command out of the message on selection', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.typeAt('fix /model| the build');
		await tick();

		const pickers = screen.component.getPickers();
		const model = pickers.availableCommands.find((c) => c.name === 'model');

		if (!model) throw new Error('model command missing');

		pickers.handleCommandSelect(model);
		await tick();
		await Promise.resolve();

		expect(screen.component.getValue()).toBe('fix the build');
		expect(screen.component.getCaretOffset()).toBe('fix '.length);
		expect(screen.component.getCalls()).toContain('openModelSelector');
	});

	it('leaves a path in the message alone', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.typeAt('look in src/lib|');
		await tick();

		expect(screen.component.getPickers().isCommandPickerOpen).toBe(false);
	});
});

describe('/cwd command in the chat input', () => {
	it('writes the current directory on a /cwd command at the caret', async () => {
		const screen = render(ChatFormPickersHarness, {
			cwd: '/home/me/projects/app',
			serverHome: '/home/me'
		});

		await tick();

		screen.component.typeAt('fix| the build');

		const pickers = screen.component.getPickers();

		pickers.handleWorkingDirectoryOpen();
		await tick();
		await Promise.resolve();

		expect(screen.component.getValue()).toBe('fix /cwd ~/projects/app the build');
		expect(screen.component.getCaretOffset()).toBe('fix /cwd ~/projects/app'.length);
		expect(pickers.isWorkingDirectoryPickerOpen).toBe(true);
		expect(pickers.isWorkingDirectoryInline).toBe(true);
		expect(pickers.workingDirectoryQuery).toBe('~/projects/app');
	});

	it('searches the path after /cwd, not the words around it', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.typeAt('fix the build|');

		const pickers = screen.component.getPickers();

		pickers.handleWorkingDirectoryOpen();
		await tick();

		expect(screen.component.getValue()).toBe('fix the build /cwd ');

		screen.component.typeAt('fix the build /cwd docs|');
		await tick();

		expect(pickers.workingDirectoryQuery).toBe('docs');
		expect(pickers.isWorkingDirectoryPickerOpen).toBe(true);

		// an edit elsewhere in the message keeps the picker and its path
		screen.component.typeAt('so,| fix the build /cwd docs');
		await tick();

		expect(pickers.workingDirectoryQuery).toBe('docs');
		expect(pickers.isWorkingDirectoryPickerOpen).toBe(true);
	});

	it('gives the message back when the picker closes', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.typeAt('fix| the build');

		const pickers = screen.component.getPickers();

		pickers.handleWorkingDirectoryOpen();
		await tick();
		screen.component.typeAt('fix /cwd docs| the build');
		pickers.handleWorkingDirectoryClose();
		await tick();

		expect(screen.component.getValue()).toBe('fix the build');
		expect(pickers.isWorkingDirectoryPickerOpen).toBe(false);
	});

	it('closes when its command is deleted, whatever else the message says', async () => {
		const screen = render(ChatFormPickersHarness, {
			cwd: '/home/me/git',
			serverHome: '/home/me'
		});

		await tick();

		const pickers = screen.component.getPickers();

		screen.component.typeAt('why does /cwd break|');
		pickers.handleWorkingDirectoryOpen();
		await tick();

		expect(screen.component.getValue()).toBe('why does /cwd break /cwd ~/git');

		// deleting ` /cwd ~/git` shifts the text by as much as the other `/cwd` is away
		screen.component.typeAt('why does /cwd break|');
		await tick();

		expect(pickers.isWorkingDirectoryPickerOpen).toBe(false);
		expect(screen.component.getValue()).toBe('why does /cwd break');
	});

	it('completes a typed /cw command where it is, on selection', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.typeAt('fix the build /cw docs|');
		await tick();

		const pickers = screen.component.getPickers();
		const cwd = pickers.availableCommands.find((c) => c.name === 'cwd');

		if (!cwd) throw new Error('cwd command missing');

		pickers.handleCommandSelect(cwd);
		await tick();

		expect(screen.component.getValue()).toBe('fix the build /cwd docs');
		expect(pickers.workingDirectoryQuery).toBe('docs');
	});

	it('tells the /cwd command from the message around it by the caret', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		const pickers = screen.component.getPickers();

		screen.component.typeAt('fix the build|');
		pickers.handleWorkingDirectoryOpen();
		await tick();

		screen.component.typeAt('fix the build /cwd docs|');
		expect(pickers.isCaretInCwdCommand()).toBe(true);

		screen.component.typeAt('fix| the build /cwd docs');
		expect(pickers.isCaretInCwdCommand()).toBe(false);
	});
});
