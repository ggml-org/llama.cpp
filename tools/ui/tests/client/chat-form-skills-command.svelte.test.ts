// Guards the `/skills` slash command contract: the command is listed only
// when the Skills service is available (per-CWD catalog slot not in error),
// and a trailing space after the `/skills` token opens the skill picker
// directly, with the typed name doubling as the fuzzy search field. A bare
// `/skills` stays on the command list, where Enter navigates to the catalog.
// Selecting a skill dispatches the trimmed name exactly once (server base
// read + durable activation); mid-typing never dispatches.

import ChatFormPickersHarness from './components/ChatFormPickersHarness.svelte';
import ChatFormSkillPickerHarness from './components/ChatFormSkillPickerHarness.svelte';
import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { SkillsService } from '$lib/services/skills.service';
import { skillsStore } from '$lib/stores/skills.svelte';
import type { ChatFormCommand, SkillCatalogEntry, SkillCatalogResponse } from '$lib/types';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

function skill(name: string, description = `${name} description`): SkillCatalogEntry {
	return {
		catalog_xml: '<skill />',
		description,
		id: `opaque-${name}`,
		instruction: { bytes: 1, lines: 1, modified_at: null, tokens: 1, tokens_estimated: false },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'global'
	};
}

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

	it('keeps a bare /skills on the command list so Enter navigates to the catalog', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(false);
		expect(pickers.skillQuery).toBe('');
		expect(pickers.isCommandPickerOpen).toBe(true);
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);

		// Explicit selection of the listed command consumes the token and
		// dispatches the no-args catalog navigation.
		pickers.handleCommandSelect(skillsCommand(pickers));
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(screen.component.getCalls().filter((c) => c === 'dispatchSkillsCommand:')).toHaveLength(
			1
		);
	});

	it('auto-opens the picker with the trimmed args as the fuzzy query while typing', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills   frontend-design');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.skillQuery).toBe('frontend-design');
		expect(pickers.isCommandPickerOpen).toBe(false);
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);
	});

	it('dispatches the named command exactly once on explicit skill selection and clears the buffer', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills   frontend-design');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(true);

		pickers.handleSkillSelect('frontend-design');
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(pickers.isSkillPickerOpen).toBe(false);
		expect(pickers.skillQuery).toBe('');
		expect(
			screen.component.getCalls().filter((c) => c === 'dispatchSkillsCommand:frontend-design')
		).toHaveLength(1);
	});

	it('keeps the retained token literal after Escape until the token changes', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills frontend-design');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(true);

		pickers.handleSkillPickerClose();
		await tick();

		// The token stays in the buffer and no command is dispatched.
		expect(screen.component.getValue()).toBe('/skills frontend-design');
		expect(pickers.isSkillPickerOpen).toBe(false);
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);

		// Typing the same token again reopens nothing: the dismissed token
		// stays literal.
		screen.component.type('/skills frontend-design');
		await tick();

		expect(pickers.isCommandPickerOpen).toBe(false);
		expect(pickers.isSkillPickerOpen).toBe(false);

		// Once the token changes, the dismissal breaks and the picker reopens.
		screen.component.type('/skills other');
		await tick();

		expect(pickers.isCommandPickerOpen).toBe(false);
		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.skillQuery).toBe('other');
	});

	it('never dispatches or clears the buffer while the token is being typed', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills add-new-model');
		await tick();

		const pickers = screen.component.getPickers();

		expect(screen.component.getValue()).toBe('/skills add-new-model');
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);
		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.skillQuery).toBe('add-new-model');
		expect(pickers.isCommandPickerOpen).toBe(false);
	});

	it('syncs skillQuery and keeps the picker open while typing /skills after auto-open', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills frontend-design');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.skillQuery).toBe('frontend-design');

		// The token doubles as the search field: typing narrows the query
		// and never re-dispatches or reopens the command picker.
		screen.component.type('/skills frontend');
		await tick();

		expect(pickers.skillQuery).toBe('frontend');
		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.isCommandPickerOpen).toBe(false);
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);
	});

	it('closes and resets the Skills picker when the token is edited away from /skills', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills frontend-design');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(true);

		// Editing the command token away from `/skills` closes the picker and
		// clears its query; the buffer itself is untouched and nothing
		// dispatches.
		screen.component.type('/cwd foo');
		await tick();

		expect(pickers.isSkillPickerOpen).toBe(false);
		expect(pickers.skillQuery).toBe('');
		expect(screen.component.getValue()).toBe('/cwd foo');
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);
	});

	it('falls back to the command list when the trailing space is deleted', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		screen.component.type('/skills ');
		await tick();

		const pickers = screen.component.getPickers();

		expect(pickers.isSkillPickerOpen).toBe(true);

		// Deleting the trailing space turns `/skills ` back into the command
		// trigger: Enter now navigates to the catalog instead.
		screen.component.type('/skills');
		await tick();

		expect(pickers.isSkillPickerOpen).toBe(false);
		expect(pickers.skillQuery).toBe('');
		expect(pickers.isCommandPickerOpen).toBe(true);
		expect(screen.component.getValue()).toBe('/skills');
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);

		pickers.handleCommandSelect(skillsCommand(pickers));
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(screen.component.getCalls().filter((c) => c === 'dispatchSkillsCommand:')).toHaveLength(
			1
		);
	});

	it('clears a stale command-dismiss snapshot on explicit skill selection so /skills discovery resumes', async () => {
		const screen = render(ChatFormPickersHarness);

		await tick();

		// Escape `/skills foo`: the live token is snapshotted as dismissed.
		screen.component.type('/skills foo');
		await tick();

		const pickers = screen.component.getPickers();

		pickers.handleSkillPickerClose();
		await tick();

		expect(screen.component.getValue()).toBe('/skills foo');

		// Re-type for a different query and activate a skill row.
		screen.component.type('/skills bar');
		await tick();

		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.skillQuery).toBe('bar');

		pickers.handleSkillSelect('bar');
		await tick();

		expect(screen.component.getValue()).toBe('');
		expect(pickers.isSkillPickerOpen).toBe(false);

		// Typing `/skills foo` again must not be treated as sticky-dismissed:
		// the successful activation cleared the stale snapshot, so the picker
		// reopens for the fresh query.
		screen.component.type('/skills foo');
		await tick();

		expect(pickers.isSkillPickerOpen).toBe(true);
		expect(pickers.skillQuery).toBe('foo');
	});
});

// Composition boundary for the `ChatFormPickers` Skills props: candidates
// arrive only as a `SkillCatalogEntry[]` prop (the ready CWD slot is derived
// in `ChatForm.svelte` and exercised by the live browser smoke test) and
// selection reports exactly the chosen skill name. The picker itself must
// never issue a catalog request.
describe('ChatFormSkillPickerHarness (ChatFormPickers Skills props)', () => {
	function rows(): HTMLElement[] {
		return Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]'));
	}

	it('renders the two catalog candidates as the displayed picker rows', async () => {
		render(ChatFormSkillPickerHarness, {
			skills: [skill('alpha'), skill('format-alpha')]
		});

		await tick();

		const rowTexts = rows().map((row) => row.textContent ?? '');

		expect(rowTexts).toHaveLength(2);
		expect(rowTexts[0]).toContain('alpha description');
		expect(rowTexts[0]).not.toContain('format-alpha');
		expect(rowTexts[1]).toContain('format-alpha description');
	});

	it('records exactly the selected candidate name', async () => {
		const screen = render(ChatFormSkillPickerHarness, {
			skills: [skill('alpha'), skill('format-alpha')]
		});

		await tick();

		rows()[1].click();
		await tick();

		expect(screen.component.getSelectedName()).toBe('format-alpha');
	});

	it('shows the no-match state and records no selection when the catalog is empty', async () => {
		const screen = render(ChatFormSkillPickerHarness, { skills: [] });

		await tick();

		expect(rows()).toHaveLength(0);
		expect(document.body.textContent).toContain('No matching skills');
		expect(screen.component.getSelectedName()).toBeNull();
	});
});

// Regression: with no conversation and no pending CWD, the form derives
// `cwd` as `null` (`activeConversation()?.cwd ?? pendingCwd()`), while the
// catalog route and store use `undefined` as the canonical no-CWD key
// (`routes/skills/+page.svelte` passes `?? undefined`). `slotFor(null)`
// misses the `undefined`-keyed ready slot, so the open picker showed no
// candidates even though the no-CWD catalog was ready. Every Skills store
// lookup in the form must canonicalize with `cwd ?? undefined`.
describe('ChatForm no-CWD ready catalog slot', () => {
	afterEach(() => {
		skillsStore.invalidate(undefined);
		vi.restoreAllMocks();
	});

	it('shows the ready no-CWD catalog candidates when no CWD is selected', async () => {
		const catalog: SkillCatalogResponse = {
			catalog_instruction_xml: '',
			diagnostics: [],
			skills: [skill('frontend-design'), skill('format-frontend-design')]
		};

		vi.spyOn(SkillsService, 'list').mockResolvedValue(catalog);

		// Seed exactly what the catalog route populates for the no-CWD
		// screen: a ready slot keyed by the canonical `undefined`.
		await skillsStore.refresh(undefined);

		const { container } = render(ChatFormTestWrapper);

		await tick();

		const textarea = container.querySelector('textarea');

		if (!(textarea instanceof HTMLTextAreaElement)) {
			throw new Error('composer textarea not rendered');
		}

		await userEvent.click(textarea);
		await userEvent.keyboard('/skills frontend-design');
		await tick();

		// Typing the token auto-opens the picker with the ready no-CWD
		// catalog, prefix match first.
		const rows = Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]'));

		expect(rows).toHaveLength(2);
		expect(rows[0].textContent).toContain('frontend-design');
		expect(rows[1].textContent).toContain('format-frontend-design');
	});
});
