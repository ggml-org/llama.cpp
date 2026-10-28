<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsSelectorList from '$lib/components/app/models/ModelsSelector/ModelsSelectorList.svelte';
	import { modelsStore } from '$lib/stores/models/index.svelte';
	import type { GroupedModelOptions, ModelItem } from '$lib/utils';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ModelsSelector'
	});

	const mockModel = (id: string, name: string, orgName?: string, tags?: string[]): ModelOption => ({
		capabilities: [],
		id,
		model: orgName ? `${orgName}/${name}` : name,
		name,
		parsedId: {
			activatedParams: null,
			modelName: name,
			orgName: orgName ?? null,
			params: null,
			quantization: null,
			raw: orgName ? `${orgName}/${name}` : name,
			tags: tags ?? []
		},
		tags
	});
</script>

<script lang="ts">
	let selectedModel = $state<string | null>(null);
	let activeId = $state<string | null>(null);

	modelsStore.favoriteModelIds = new Set(['qwen2.5-7b', 'llama3.2-3b']);

	const favoriteModels: ModelItem[] = [
		{ option: mockModel('qwen2.5-7b', 'Qwen2.5-7B-Instruct', 'Qwen') },
		{ option: mockModel('llama3.2-3b', 'Llama-3.2-3B-Instruct', 'meta') }
	];

	const availableModels: ModelItem[] = [
		{
			option: mockModel('deepseek-coder-6.7b', 'DeepSeek-Coder-6.7B', 'deepseek', ['coding'])
		},
		{ option: mockModel('gemma-2-9b', 'Gemma-2-9B-IT', 'google') },
		{ option: mockModel('phi-3-mini', 'Phi-3-mini-4k', 'microsoft') },
		{ option: mockModel('codellama-7b', 'CodeLlama-7B', 'codellama', ['coding']) },
		{ option: mockModel('neural-chat-7b', 'Neural-Chat-7B-v3-3', 'intel') }
	];

	const groupedOptions: GroupedModelOptions = {
		available: [
			{
				items: [availableModels[0]],
				orgName: 'deepseek'
			},
			{
				items: [availableModels[1]],
				orgName: 'google'
			},
			{
				items: [availableModels[2]],
				orgName: 'microsoft'
			},
			{
				items: [availableModels[3]],
				orgName: 'codellama'
			},
			{
				items: [availableModels[4]],
				orgName: 'intel'
			}
		]
	};

	function handleSelect(modelId: string) {
		const opt = [...favoriteModels, ...availableModels].find((m) => m.option.id === modelId);

		if (opt) {
			selectedModel = opt.option.model;
			activeId = modelId;
		}
	}
</script>

<Story name="Default">
	<div class="w-80 rounded-xl border matte-float p-1.5">
		<ModelsSelectorList
			{activeId}
			currentModel={selectedModel}
			favorites={favoriteModels}
			groups={groupedOptions}
			onSelect={handleSelect}
		/>
	</div>
</Story>
