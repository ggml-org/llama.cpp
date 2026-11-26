<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsSelectorSheet from '#lib/components/app/models/ModelsSelector/ModelsSelectorSheet.svelte';
	import { MODELS_SHEET_TEXT } from '#lib/constants/index.ts';
	import type { GroupedModelOptions, ModelItem } from '#lib/utils/index.ts';
	import { screen } from 'storybook/test';

	// a phone's model picker: a short sheet with the models in reach (the chat's, the loaded,
	// favorite and recent ones) and a row to all of them, a page of groups. The search stays at
	// the sheet's foot, and holding a row opens its menu
	const { Story } = defineMeta({
		component: ModelsSelectorSheet,
		globals: { viewport: { isRotated: false, value: 'mobile2' } },
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Models/ModelsSelectorSheet'
	});

	const model = (name: string, orgName: string): ModelItem => ({
		option: {
			capabilities: [],
			id: `${orgName}/${name}`,
			model: `${orgName}/${name}`,
			name
		}
	});

	const QWEN = model('Qwen3-8B', 'Qwen');
	const GEMMA = model('gemma-3n-E4B-it', 'google');
	const LLAMA = model('Llama-3.2-3B-Instruct', 'meta');
	const PHI = model('Phi-4-mini', 'microsoft');
	const REMOVED_MODEL = 'acme/Removed-7B';

	const groups: GroupedModelOptions = {
		available: [
			{ items: [LLAMA], orgName: 'meta' },
			{ items: [PHI], orgName: 'microsoft' }
		]
	};

	const args = {
		activeId: QWEN.option.id,
		favoriteItems: [GEMMA],
		groups,
		loadedItems: [QWEN],
		open: true,
		quickItems: [QWEN, GEMMA]
	};
</script>

<Story {args} name="Models in reach" />

<Story
	{args}
	name="All models"
	play={async ({ userEvent }) => {
		await userEvent.click(
			await screen.findByRole('button', { name: MODELS_SHEET_TEXT.ALL_MODELS })
		);
		await screen.findByRole('heading', { name: MODELS_SHEET_TEXT.LOADED });
	}}
/>

<Story args={{ ...args, searchTerm: 'phi' }} name="Search results" />

<Story
	args={{ ...args, activeId: null, quickItems: [GEMMA], unavailableModel: REMOVED_MODEL }}
	name="Chat model not available"
/>

<Story args={{ isEmpty: true, open: true, searchTerm: 'nothing' }} name="No models found" />
