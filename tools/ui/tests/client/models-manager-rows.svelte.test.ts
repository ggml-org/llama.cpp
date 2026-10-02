// Guards the row semantics of the manager table: each row's primary control is a
// real button, so no control of the row sits inside another one, and a row that
// discloses its quants says so with aria-expanded.

import ModelsManagerRowWrapper from './components/ModelsManagerRowWrapper.svelte';
import ModelsManagerRepoRow from '$lib/components/app/models/ModelsManager/ModelsManagerRepoRow.svelte';
import type { ModelQuantGroup } from '$lib/components/app/models/ModelsManager/utils';
import { ModelGroupKind } from '$lib/enums';
import type { ModelOption } from '$lib/types/models';
import { beforeEach, describe, expect, it } from 'vitest';
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
/** Every control a row can hold, so the test can tell a sibling from a nested one. */
const CONTROLS = 'button, a[href], input, select, textarea';

describe('manager model row', () => {
	let selected: ModelOption[] = [];

	beforeEach(() => {
		selected = [];
	});

	function row(selectedModel: boolean) {
		return render(ModelsManagerRowWrapper, {
			onSelect: (picked: ModelOption) => selected.push(picked),
			option,
			selected: selectedModel
		});
	}

	it('selects the model from a button that holds no other control', async () => {
		const screen = await row(false);
		const select = screen.container.querySelector('button');

		expect(select).not.toBeNull();
		expect(select?.querySelectorAll(CONTROLS)).toHaveLength(0);

		await screen.getByRole('button', { name: ROW_NAME }).click();

		expect(selected).toEqual([option]);
	});

	it('marks the selected row for assistive technology', async () => {
		const screen = await row(true);

		await expect
			.element(screen.getByRole('button', { name: ROW_NAME }))
			.toHaveAttribute('aria-current', 'true');
	});

	it('keeps the load and actions controls beside the select control', async () => {
		const screen = await row(false);
		const select = screen.container.querySelector('button');
		const actions = screen.getByRole('button', { name: 'Model actions' }).element();

		expect(select?.contains(actions)).toBe(false);
		expect(select?.querySelectorAll(CONTROLS)).toHaveLength(0);
	});
});

describe('manager repo row', () => {
	it('discloses the quants from an expanded button', async () => {
		const screen = await render(ModelsManagerRepoRow, {
			entry,
			expanded: true,
			onToggle: () => {}
		});

		await expect
			.element(screen.getByRole('button', { name: /2 quants available/ }))
			.toHaveAttribute('aria-expanded', 'true');
	});
});
