// Guards the model-consent card's Skills rendering: when a Skills consent
// pause carries safe server identity facts (`SkillConsentInfo`), the card
// shows them (name, scope · provider, optional resource path) without
// touching the generic label path; without a skill the card renders the
// established generic text unchanged.

import ChatMessageActionCardPermissionRequest from '$lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardPermissionRequest.svelte';
import { ToolPermissionDecision } from '$lib/enums';
import type { SkillConsentInfo } from '$lib/types';
import { tick } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

async function renderCard(skill?: SkillConsentInfo) {
	const { container } = render(ChatMessageActionCardPermissionRequest, {
		onDecision: vi.fn(),
		serverLabel: 'llama-server',
		skill,
		toolName: 'read_skill'
	});

	await tick();

	return container;
}

function textOf(container: HTMLElement): string {
	return (container.textContent ?? '').replace(/\s+/g, ' ').trim();
}

describe('permission request card skill identity', () => {
	it('shows the safe skill identity for a base consent pause', async () => {
		const container = await renderCard({
			name: 'add-new-model',
			scope: 'project',
			provider: 'agents'
		});

		const text = textOf(container);

		expect(text).toContain('Allow use of read_skill from llama-server?');
		expect(text).toContain('Skill: add-new-model (project · agents)');
		expect(text).not.toContain('resource:');
	});

	it('shows the requested relative path for a resource consent pause', async () => {
		const container = await renderCard({
			name: 'add-new-model',
			scope: 'project',
			provider: 'agents',
			path: 'refs/DETAILS.md'
		});

		expect(textOf(container)).toContain('Skill: add-new-model (project · agents)— resource: refs/DETAILS.md');
	});

	it('renders the generic prompt unchanged when no skill identity is present', async () => {
		const container = await renderCard();

		const text = textOf(container);

		expect(text).toContain('Allow use of read_skill from llama-server?');
		expect(text).not.toContain('Skill:');
	});
});
