// Guards which org a model avatar shows: a row that names neither flag follows the
// family grouping (grouped lists leave the base org to the family heading), while a
// caller that names one flag keeps control of what the avatar shows.

import ModelAvatar from '$lib/components/app/models/ModelAvatar.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ModelOption } from '$lib/types/models';
import { beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

// the base-model tag resolves the base org without a Hub lookup
const option: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B-GGUF:Q4_K_M',
	model: 'org/Qwen3-8B-GGUF:Q4_K_M',
	name: 'Qwen3-8B',
	tags: ['base_model:Qwen/Qwen3-8B']
};
/** The main avatar plus the quantizer badge, when the badge differs from it. */
const avatars = (container: HTMLElement) => container.querySelectorAll('img').length;

function setGrouping(grouped: boolean) {
	settingsStore.updateConfig(SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY, grouped);
	settingsStore.updateConfig(SETTINGS_KEYS.USE_HUGGING_FACE_HUB, true);
}

beforeEach(() => {
	setGrouping(false);
});

describe('model avatar orgs', () => {
	it('shows the base org with the quantizer badge when the lists are flat', async () => {
		const screen = render(ModelAvatar, { option });

		expect(avatars(screen.container)).toBe(2);
	});

	it('leaves the base org to the family heading when the lists are grouped', async () => {
		setGrouping(true);

		const screen = render(ModelAvatar, { option });

		expect(avatars(screen.container)).toBe(1);
	});

	it('keeps the base org of a caller that names it, grouped lists or not', async () => {
		setGrouping(true);

		const screen = render(ModelAvatar, { option, showBaseModelAvatar: true });

		expect(avatars(screen.container)).toBe(2);
	});

	it('keeps the repo org of a caller that names it, flat lists or not', async () => {
		const screen = render(ModelAvatar, { option, showRepoOrgAvatar: true });

		expect(avatars(screen.container)).toBe(1);
	});
});
