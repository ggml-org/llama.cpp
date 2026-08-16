import SkillCatalogList from '$lib/components/app/skills/SkillCatalogList.svelte';
import type { SkillCatalogEntry } from '$lib/types';
import { describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

function makeEntry(name: string, overrides: Partial<SkillCatalogEntry> = {}): SkillCatalogEntry {
	return {
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project',
		...overrides
	};
}

function bodyText(): string {
	return document.body.textContent ?? '';
}

describe('SkillCatalogList manual-only badge', () => {
	it('renders a Manual only badge for flagged entries', async () => {
		const screen = await render(SkillCatalogList, {
			props: {
				entries: [makeEntry('manual', { disable_model_invocation: true })],
				selectedId: null,
				open: false,
				onSelect: vi.fn()
			}
		});

		await expect.element(screen.getByText('Manual only')).toBeInTheDocument();
	});

	it('omits the badge for model-visible entries', async () => {
		await render(SkillCatalogList, {
			props: {
				entries: [
					makeEntry('normal'),
					makeEntry('legacy', { disable_model_invocation: false }),
					makeEntry('older', { disable_model_invocation: undefined })
				],
				selectedId: null,
				open: false,
				onSelect: vi.fn()
			}
		});

		expect(bodyText()).not.toContain('Manual only');
	});
});
