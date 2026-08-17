// Form-level wake contract: selecting a skill dispatches the durable
// activation and a successful outcome wakes the agentic loop through
// chatStore.runTurnFromLeaf. Not-found and unavailable outcomes never wake.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { dispatchSkillActivation } from '$lib/services/skill-command.service';
import { chatStore } from '$lib/stores';
import { skillsStore } from '$lib/stores/skills.svelte';
import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
import type { SkillCatalogEntry } from '$lib/types';
import type { SkillCatalogSlot } from '$lib/stores/skills.svelte';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

vi.mock('$lib/services/skill-command.service', () => ({ dispatchSkillActivation: vi.fn() }));
vi.mock('$lib/stores/skills.svelte', () => ({ skillsStore: { slotFor: vi.fn() } }));

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

function readySlot(entries: SkillCatalogEntry[]): SkillCatalogSlot {
	return {
		catalog: { catalog_instruction_xml: '', diagnostics: [], skills: entries },
		cwd: undefined,
		generation: 1,
		status: 'ready'
	};
}

async function selectSkill(name: string) {
	const { container } = render(ChatFormTestWrapper);

	await tick();

	const textarea = container.querySelector('textarea');

	if (!(textarea instanceof HTMLTextAreaElement)) throw new Error('textarea not rendered');

	await userEvent.click(textarea);
	await userEvent.keyboard(`/skills ${name}`);
	await tick();
	// The sole candidate is pre-highlighted on open; Enter selects it.
	await userEvent.keyboard('{Enter}');
	await tick();
}

beforeEach(() => {
	skillAvailabilityStore.setEnabled('opaque-frontend-design', true);
	skillAvailabilityStore.setEnabled('opaque-disabled', true);
	skillAvailabilityStore.setEnabled('opaque-enabled-a', true);
	skillAvailabilityStore.setEnabled('opaque-enabled-b', true);
	vi.mocked(skillsStore.slotFor).mockReturnValue(readySlot([skill('frontend-design')]));
	vi.mocked(dispatchSkillActivation).mockReset();
	vi.mocked(dispatchSkillActivation).mockResolvedValue({ ok: true, created: true });
});

afterEach(() => {
	vi.restoreAllMocks();
});

describe('/skills <name> wake', () => {
	it('wakes the agentic loop after a successful activation', async () => {
		const runTurn = vi.spyOn(chatStore, 'runTurnFromLeaf').mockResolvedValue();

		await selectSkill('frontend-design');

		await vi.waitFor(() =>
			expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalledWith('frontend-design')
		);
		await vi.waitFor(() => expect(runTurn).toHaveBeenCalledTimes(1));
	});

	it('omits disabled picker entries and preserves enabled server order', async () => {
		vi.mocked(skillsStore.slotFor).mockReturnValue(
			readySlot([skill('disabled'), skill('enabled-b'), skill('enabled-a')])
		);
		skillAvailabilityStore.setEnabled('opaque-disabled', false);

		const { container } = render(ChatFormTestWrapper);
		await tick();

		const textarea = container.querySelector('textarea');

		if (!(textarea instanceof HTMLTextAreaElement)) throw new Error('textarea not rendered');

		await userEvent.click(textarea);
		await userEvent.keyboard('/skills ');
		await tick();

		const rowText = Array.from(document.querySelectorAll<HTMLElement>('[data-picker-index]')).map(
			(row) => row.textContent ?? ''
		);

		expect(rowText).toHaveLength(2);
		expect(rowText[0]).toContain('enabled-b');
		expect(rowText[1]).toContain('enabled-a');
		expect(rowText.join(' ')).not.toContain('disabled');
	});

	it.each(['disabled', 'not-found', 'unavailable', 'persistence-failed'] as const)(
		'does not wake when the activation %s',
		async (reason) => {
			const runTurn = vi.spyOn(chatStore, 'runTurnFromLeaf').mockResolvedValue();

			vi.mocked(dispatchSkillActivation).mockResolvedValue({ ok: false, reason });

			await selectSkill('frontend-design');

			await vi.waitFor(() => expect(vi.mocked(dispatchSkillActivation)).toHaveBeenCalled());
			expect(runTurn).not.toHaveBeenCalled();
		}
	);
});