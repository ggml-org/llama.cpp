<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsManagerModelRow from '$lib/components/app/models/ModelsManager/ModelsManagerModelRow.svelte';
	import ModelsManagerRepoRow from '$lib/components/app/models/ModelsManager/ModelsManagerRepoRow.svelte';
	import type { ModelQuantGroup } from '$lib/components/app/models/ModelsManager/utils';
	import { MODEL_ROW_GRID_CLASS } from '$lib/constants';
	import { ModelGroupKind } from '$lib/enums';
	import type { ModelOption } from '$lib/types/models';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Components/ModelsManager/Accessibility'
	});

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
		quants: [option]
	};

	const ROW_NAME = /org\/Qwen3\s+8B/;
</script>

<!-- The row itself is not a control: its name cell takes focus and the actions
     control is the next tab stop, so a keyboard user never lands on a button that
     holds another button. -->
<Story
	name="RowTabStops"
	play={async ({ canvas, userEvent }) => {
		const select = await canvas.findByRole('button', { name: ROW_NAME });

		select.focus();
		await userEvent.tab();

		await expect(await canvas.findByRole('button', { name: 'Model actions' })).toHaveFocus();
	}}
>
	<div class={MODEL_ROW_GRID_CLASS + ' w-[40rem]'}>
		<ModelsManagerModelRow
			isFavorite={() => false}
			onDelete={() => {}}
			onSelect={() => {}}
			{option}
			selected={false}
		/>
	</div>
</Story>

<!-- A repo row discloses its quants, so its control carries aria-expanded and
     answers to the keyboard without a keydown handler of its own. -->
<Story
	name="RepoRowDisclosure"
	play={async ({ canvas, userEvent }) => {
		const toggle = await canvas.findByRole('button', { name: /1 quants available/ });

		await expect(toggle).toHaveAttribute('aria-expanded', 'false');

		toggle.focus();
		await userEvent.keyboard('{Enter}');
	}}
>
	<div class={MODEL_ROW_GRID_CLASS + ' w-[40rem]'}>
		<ModelsManagerRepoRow {entry} expanded={false} onToggle={() => {}} />
	</div>
</Story>
