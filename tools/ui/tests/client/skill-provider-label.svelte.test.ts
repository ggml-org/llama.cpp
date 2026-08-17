// Guards the shared provider label and generic tooltip across Skills surfaces.

import SkillProviderLabel from '$lib/components/app/skills/SkillProviderLabel.svelte';
import { GENERIC_SKILL_PROVIDER_TOOLTIP } from '$lib/components/app/skills/skill-provider-presentation';
import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

describe('SkillProviderLabel', () => {
	it('maps the agents API provider value to the generic label', async () => {
		const { container } = render(SkillProviderLabel, { props: { provider: 'agents' } });

		expect(container.textContent).toContain('generic');
		expect(container.textContent).not.toContain('agents');
	});

	it('exposes the exact provider-agnostic tooltip on keyboard focus', async () => {
		const { container } = render(SkillProviderLabel, { props: { provider: 'agents' } });

		const trigger = container.querySelector('[data-slot="tooltip-trigger"]');

		expect(trigger).not.toBeNull();

		trigger!.dispatchEvent(new FocusEvent('focus', { bubbles: true }));

		await vi.waitFor(() => {
			expect(document.body.textContent).toContain(GENERIC_SKILL_PROVIDER_TOOLTIP);
		});
	});

	it('renders a non-agents provider unchanged without a provider-agnostic tooltip', async () => {
		const { container } = render(SkillProviderLabel, { props: { provider: 'claude' } });

		expect(container.textContent).toContain('claude');
		expect(container.textContent).not.toContain('generic');
		expect(container.querySelector('[data-slot="tooltip-trigger"]')).toBeNull();

		// No hidden tooltip is wired for non-agents providers.
		const bodyBefore = document.body.textContent ?? '';
		expect(bodyBefore).not.toContain(GENERIC_SKILL_PROVIDER_TOOLTIP);
	});
});