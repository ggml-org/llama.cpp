// Guards the `/skills` slash command contract: the command is listed only
// when the Skills service is available (per-CWD catalog slot not in error),
// explicit selection consumes the token and forwards the trimmed args to
// the skills dispatch (catalog navigation for no args, server base read +
// durable activation for a name), and mid-typing never dispatches — the
// same explicit-selection rule the other slash commands follow.

import ChatFormPickersHarness from './components/ChatFormPickersHarness.svelte';
import type { ChatFormCommand } from '$lib/types';
import { tick } from 'svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

function skillsCommand(pickers: { availableCommands: ChatFormCommand[] }): ChatFormCommand {
	const command = pickers.availableCommands.find((c) => c.name === 'skills');

	if (!command) throw new Error('skills command missing');

	return command;
}

describe('/skills command', () => {
	it('is listed and enabled when the Skills service is available', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		expect(skillsCommand(screen.component.getPickers()).disabled).toBe(false);
	});

	it('is disabled when the Skills service is unavailable', async () => {
		const screen = render(ChatFormPickersHarness, { hasSkills: false });

		await tick();

		expect(skillsCommand(screen.component.getPickers()).disabled).toBe(true);
	});

	it('dispatches /skills with no args (catalog navigation) and consumes the token', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills');
		await tick();

		const pickers = screen.component.getPickers();

		pickers.handleCommandSelect(skillsCommand(pickers));
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(screen.component.getCalls()).toContain('dispatchSkillsCommand:');
		expect(pickers.isCommandPickerOpen).toBe(false);
	});

	it('forwards the trimmed token args to the skills activation dispatch', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills   add-new-model');
		await tick();

		const pickers = screen.component.getPickers();

		pickers.handleCommandSelect(skillsCommand(pickers));
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(screen.component.getCalls()).toContain('dispatchSkillsCommand:add-new-model');
		expect(pickers.isCommandPickerOpen).toBe(false);
	});

	it('does not dispatch or clear the buffer while the token is still being typed', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills add-new-model');
		await tick();

		const pickers = screen.component.getPickers();

		expect(screen.component.getValue()).toBe('/skills add-new-model');
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(false);
		expect(pickers.isCommandPickerOpen).toBe(true);
	});
});
