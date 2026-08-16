// Guards the `/skills` slash command contract: the command is listed only
// when the Skills service is available (per-CWD catalog slot not in error),
// and a trailing space after the `/skills` token opens the skill picker
// directly, with the typed name doubling as the fuzzy search field. A bare
// `/skills` stays on the command list, where Enter navigates to the catalog.
// Selecting a skill dispatches the trimmed name exactly once (server base
// read + durable activation); mid-typing never dispatches.
//
// The picker itself guards the safe catalog-name contract: candidates are
// ordered prefix-first then substring in source catalog order, rows expose
// only the safe display facts (name, description, scope, provider) - never
// the opaque id, instruction facts, resources, or raw catalog XML - and
// keyboard / pointer selection reports the exact skill name. The picker
// receives everything via props; it must never fetch, call SkillsService,
// resolve names, or render XML itself.

import ChatFormSkillPicker from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormSkillPicker.svelte';
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

function keydown(key: string): KeyboardEvent {
	return new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key });
}

function renderPicker(query: string) {
	const onClose = vi.fn();
	const onSelect = vi.fn();
	const screen = render(ChatFormSkillPicker, {
		isOpen: true,
		onClose,
		onSelect,
		query,
		skills: [skill('alpha'), skill('format-alpha'), skill('beta')]
	});

	return { onClose, onSelect, screen };
}

function rows(): HTMLElement[] {
	return Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]'));
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

		// The token doubles as the search field: typing further narrows the
		// query and never re-dispatches or reopens the command picker.
		screen.component.type('/skills frontend');
		await tick();

		expect(pickers.skillQuery).toBe('frontend');
		expect(pickers.isSkillPickerOpen).toBe(true);
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

		// Mid-typing the token never dispatches and never clears the buffer.
		expect(screen.component.getValue()).toBe('/skills   frontend-design');
		expect(screen.component.getCalls().some((c) => c.startsWith('dispatchSkillsCommand'))).toBe(
			false
		);

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

describe('ChatFormSkillPicker', () => {
	it('lists prefix matches before substring matches in source order', async () => {
		renderPicker('alpha');

		await tick();

		const rowTexts = rows().map((row) => row.textContent ?? '');

		expect(rowTexts).toHaveLength(2);
		expect(rowTexts[0]).toContain('alpha description');
		expect(rowTexts[0]).not.toContain('format-alpha');
		expect(rowTexts[1]).toContain('format-alpha');
	});

	it('renders only safe display facts, never id, instruction, or catalog XML', async () => {
		renderPicker('alpha');

		await tick();

		const bodyText = document.body.textContent ?? '';

		expect(bodyText).toContain('alpha');
		expect(bodyText).toContain('format-alpha');
		expect(bodyText).not.toContain('opaque-alpha');
		expect(bodyText).not.toContain('opaque-format-alpha');
		expect(bodyText).not.toContain('<skill />');
	});

	it('selects the exact name with ArrowDown + Enter and closes', async () => {
		const { onClose, onSelect, screen } = renderPicker('alpha');

		await tick();

		// The first candidate is pre-highlighted on open; one ArrowDown
		// moves to the second candidate (prefix first: alpha, then
		// format-alpha) and Enter reports the exact name.
		screen.component.handleKeydown(keydown('ArrowDown'));
		screen.component.handleKeydown(keydown('Enter'));
		await tick();

		expect(onSelect).toHaveBeenCalledTimes(1);
		expect(onSelect).toHaveBeenCalledWith('format-alpha');
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('selects the exact name on row click', async () => {
		const { onClose, onSelect } = renderPicker('alpha');

		await tick();

		rows()[1].click();
		await tick();

		expect(onSelect).toHaveBeenCalledTimes(1);
		expect(onSelect).toHaveBeenCalledWith('format-alpha');
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it('closes on Escape without selecting', async () => {
		const { onClose, onSelect, screen } = renderPicker('alpha');

		await tick();

		screen.component.handleKeydown(keydown('Escape'));
		await tick();

		expect(onClose).toHaveBeenCalledTimes(1);
		expect(onSelect).not.toHaveBeenCalled();
	});

	it('shows the empty message and does not select for a no-match query', async () => {
		const { onClose, onSelect, screen } = renderPicker('missing');

		await tick();

		expect(rows()).toHaveLength(0);
		expect(document.body.textContent).toContain('No matching skills');

		// Enter is not consumed without a selectable row, so the caller's
		// Enter-to-submit still runs; nothing is selected or closed.
		expect(screen.component.handleKeydown(keydown('Enter'))).toBe(false);
		expect(onSelect).not.toHaveBeenCalled();
		expect(onClose).not.toHaveBeenCalled();
	});
});

// Composition boundary for the `ChatFormPickers` Skills props: candidates
// arrive only as a `SkillCatalogEntry[]` prop (the ready CWD slot is derived
// in `ChatForm.svelte` and exercised by the live browser smoke test) and
// selection reports exactly the chosen skill name. The picker itself must
// never issue a catalog request.
describe('ChatFormSkillPickerHarness (ChatFormPickers Skills props)', () => {
	it('renders the controlled candidates and shows the no-match state when the catalog is empty', async () => {
		const empty = render(ChatFormSkillPickerHarness, { skills: [] });

		await tick();

		expect(rows()).toHaveLength(0);
		expect(document.body.textContent).toContain('No matching skills');
		expect(empty.component.getSelectedName()).toBeNull();

		const screen = render(ChatFormSkillPickerHarness, {
			skills: [skill('alpha'), skill('format-alpha')]
		});

		await tick();

		expect(rows()).toHaveLength(2);
		expect(screen.component.getSelectedName()).toBeNull();
		expect(rows()[0].textContent).toContain('alpha description');
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
		const candidateRows = Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]'));

		expect(candidateRows).toHaveLength(2);
		expect(candidateRows[0].textContent).toContain('frontend-design');
		expect(candidateRows[1].textContent).toContain('format-frontend-design');
	});
});