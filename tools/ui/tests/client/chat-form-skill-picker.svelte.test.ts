// Guards the safe catalog-name picker contract: candidates are ordered
// prefix-first then substring in source catalog order, rows expose only the
// safe display facts (name, description, scope, provider) - never the opaque
// id, instruction facts, resources, or raw catalog XML - and keyboard /
// pointer selection reports the exact skill name. The picker receives
// everything via props; it must never fetch, call SkillsService, resolve
// names, or render XML itself.

import ChatFormSkillPicker from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormSkillPicker.svelte';
import type { SkillCatalogEntry } from '$lib/types';
import { tick } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
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

	it('renders multiline descriptions as one paragraph', async () => {
		const onClose = vi.fn();
		const onSelect = vi.fn();

		render(ChatFormSkillPicker, {
			isOpen: true,
			onClose,
			onSelect,
			query: 'alpha',
			skills: [skill('alpha', 'This is line one.\n  This is line two.\n\nThis is the final line.')]
		});

		await tick();

		expect(rows()[0].textContent).toContain(
			'This is line one. This is line two. This is the final line.'
		);
	});

	it('selects the exact name with ArrowDown + Enter and closes', async () => {
		const { onClose, onSelect, screen } = renderPicker('alpha');

		await tick();

		// The first candidate is pre-highlighted on open; one ArrowDown
		// moves to the second candidate (prefix first: alpha, then
		// format-alpha) and Enter reports the exact name.
		expect(screen.component.handleKeydown(keydown('ArrowDown'))).toBe(true);
		expect(screen.component.handleKeydown(keydown('Enter'))).toBe(true);
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

		expect(screen.component.handleKeydown(keydown('Escape'))).toBe(true);
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
