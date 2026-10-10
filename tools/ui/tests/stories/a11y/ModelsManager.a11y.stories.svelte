<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsManagerModelRow from '$lib/components/app/models/ModelsManager/ModelsManagerModelRow.svelte';
	import { MODEL_ROW_GRID_CLASS } from '$lib/constants';
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

	const ROW_NAME = /org\/Qwen3\s+8B/;
</script>

<script lang="ts">
	let selected = $state(false);
</script>

<!-- The row answers Enter and Space on its own, so a keyboard user selects a model
     without reaching for the pointer. -->
<Story
	name="RowKeyboardSelect"
	play={async ({ canvas, userEvent }) => {
		const row = await canvas.findByRole('button', { name: ROW_NAME });

		row.focus();
		await userEvent.keyboard('{Enter}');

		await expect(row).toHaveAttribute('aria-pressed', 'true');
	}}
>
	<div class={MODEL_ROW_GRID_CLASS + ' w-[40rem]'}>
		<ModelsManagerModelRow
			isFavorite={() => false}
			onDelete={() => {}}
			onSelect={() => (selected = true)}
			{option}
			{selected}
		/>
	</div>
</Story>
