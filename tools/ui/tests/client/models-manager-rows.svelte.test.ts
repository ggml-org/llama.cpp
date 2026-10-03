// Guards the row behaviour of the manager table: a row selects its model with a
// click or from the keyboard, and a repo row toggles its quants the same way.

import ModelsManagerRowWrapper from './components/ModelsManagerRowWrapper.svelte';
import ModelsManagerRepoRow from '$lib/components/app/models/ModelsManager/ModelsManagerRepoRow.svelte';
import type { ModelQuantGroup } from '$lib/components/app/models/ModelsManager/utils';
import { ModelGroupKind, ServerRole } from '$lib/enums';
import { serverStore } from '$lib/stores/server.svelte';
import type { ModelOption } from '$lib/types/models';
import { beforeEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const option: ModelOption = {
	capabilities: [],
	id: 'org/Qwen3-8B:Q4_K_M',
	model: 'org/Qwen3-8B:Q4_K_M',
	name: 'Qwen3-8B'
};
const entry: ModelQuantGroup = {
	base: option,
	key: 'org/Qwen3-8B',
	kind: ModelGroupKind.QUANTS,
	quants: [
		option,
		{ ...option, id: 'org/Qwen3-8B:Q8_0', model: 'org/Qwen3-8B:Q8_0', name: 'Qwen3-8B-Q8_0' }
	]
};
/** The name the row carries once the model id is split into its badges. */
const ROW_NAME = /org\/Qwen3\s+8B/;

describe('manager model row', () => {
	let selected: ModelOption[] = [];

	beforeEach(() => {
		selected = [];
		// the row only offers a load control on a router server
		serverStore.role = ServerRole.ROUTER;
	});

	function row() {
		return render(ModelsManagerRowWrapper, {
			onSelect: (picked: ModelOption) => selected.push(picked),
			option,
			selected: false
		});
	}

	it('selects the model from the row', async () => {
		const screen = await row();

		await screen.getByRole('button', { name: ROW_NAME }).click();

		expect(selected).toEqual([option]);
	});

	it('selects the model from the keyboard', async () => {
		const screen = await row();
		const target = screen.getByRole('button', { name: ROW_NAME }).element();

		target.focus();
		await userEvent.keyboard('{Enter}');

		expect(selected).toEqual([option]);
	});

	it('keeps the load and actions controls of the row', async () => {
		const screen = await row();

		await expect
			.element(screen.getByRole('button', { exact: true, name: 'Model actions' }))
			.toBeVisible();
		expect(screen.container.querySelector('[aria-label="Load model"]')).not.toBeNull();
	});
});

describe('manager repo row', () => {
	it('toggles the quants from the row', async () => {
		let toggles = 0;

		const screen = await render(ModelsManagerRepoRow, {
			entry,
			expanded: true,
			onToggle: () => (toggles += 1)
		});

		await screen.getByRole('button', { name: /2 quants available/ }).click();

		expect(toggles).toBe(1);
	});
});
