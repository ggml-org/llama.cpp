<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsSelectorList from '$lib/components/app/models/ModelsSelectorList.svelte';
	import ModelsSelectorOption from '$lib/components/app/models/ModelsSelectorOption.svelte';
	import type { GroupedModelOptions, ModelItem } from '$lib/components/app/models/utils';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { ServerModelStatus } from '$lib/enums';

	const { Story } = defineMeta({
		title: 'Components/ModelsSelector',
		parameters: {
			layout: 'centered'
		}
	});

	const mockModel = (id: string, name: string, orgName?: string, tags?: string[]): ModelOption => ({
		id,
		name,
		model: orgName ? `${orgName}/${name}` : name,
		capabilities: [],
		parsedId: {
			raw: orgName ? `${orgName}/${name}` : name,
			orgName: orgName ?? null,
			modelName: name,
			params: null,
			activatedParams: null,
			quantization: null,
			tags: tags ?? []
		},
		tags
	});

	const mockRouterEntry = (modelName: string, status: ServerModelStatus): ApiModelDataEntry => ({
		id: modelName,
		object: 'model',
		owned_by: 'llamacpp',
		created: Date.now(),
		in_cache: true,
		path: `/models/${modelName}`,
		status: { value: status }
	});
</script>

<script lang="ts">
	let selectedModel = $state<string | null>(null);
	let activeId = $state<string | null>(null);

	function mockModelsStore() {
		modelsStore.favoriteModelIds = new Set(['qwen2.5-7b', 'llama3.2-3b']);

		// Mock router models with various statuses for ModelLoadedStates story
		modelsStore.routerModels = [
			mockRouterEntry('meta/Model (loading)', ServerModelStatus.LOADING),
			mockRouterEntry('meta/Model (loaded)', ServerModelStatus.LOADED),
			mockRouterEntry('meta/Model (sleeping)', ServerModelStatus.SLEEPING),
			mockRouterEntry('meta/Model (failed)', ServerModelStatus.FAILED)
		];
	}

	mockModelsStore();

	const loadedModels: ModelItem[] = [
		{ option: mockModel('llama3.1-8b', 'Llama-3.1-8B-Instruct', 'meta'), flatIndex: 0 },
		{ option: mockModel('mistral-7b', 'Mistral-7B-v0.3', 'mistralai'), flatIndex: 1 }
	];

	const favoriteModels: ModelItem[] = [
		{ option: mockModel('qwen2.5-7b', 'Qwen2.5-7B-Instruct', 'Qwen'), flatIndex: 2 },
		{ option: mockModel('llama3.2-3b', 'Llama-3.2-3B-Instruct', 'meta'), flatIndex: 3 }
	];

	const availableModels: ModelItem[] = [
		{
			option: mockModel('deepseek-coder-6.7b', 'DeepSeek-Coder-6.7B', 'deepseek', ['coding']),
			flatIndex: 4
		},
		{ option: mockModel('gemma-2-9b', 'Gemma-2-9B-IT', 'google'), flatIndex: 5 },
		{ option: mockModel('phi-3-mini', 'Phi-3-mini-4k', 'microsoft'), flatIndex: 6 },
		{ option: mockModel('codellama-7b', 'CodeLlama-7B', 'codellama', ['coding']), flatIndex: 7 },
		{ option: mockModel('neural-chat-7b', 'Neural-Chat-7B-v3-3', 'intel'), flatIndex: 8 }
	];

	const groupedOptions: GroupedModelOptions = {
		loaded: loadedModels,
		favorites: favoriteModels,
		available: [
			{
				orgName: 'deepseek',
				items: [availableModels[0]]
			},
			{
				orgName: 'google',
				items: [availableModels[1]]
			},
			{
				orgName: 'microsoft',
				items: [availableModels[2]]
			},
			{
				orgName: 'codellama',
				items: [availableModels[3]]
			},
			{
				orgName: 'intel',
				items: [availableModels[4]]
			}
		]
	};

	function handleSelect(modelId: string) {
		const opt = [...loadedModels, ...favoriteModels, ...availableModels].find(
			(m) => m.option.id === modelId
		);
		if (opt) {
			selectedModel = opt.option.model;
			activeId = modelId;
		}
	}
</script>

<Story name="List">
	<div class="w-80 rounded-lg border border-border bg-popover p-2 shadow-md">
		<ModelsSelectorList
			groups={groupedOptions}
			currentModel={selectedModel}
			{activeId}
			onSelect={handleSelect}
			onInfoClick={(modelName) => console.log('Info clicked:', modelName)}
		/>
	</div>
</Story>

<Story name="SingleLoaded">
	<div class="w-80 rounded-lg border border-border bg-popover p-2 shadow-md">
		<ModelsSelectorList
			groups={{
				loaded: [loadedModels[0]],
				favorites: [],
				available: []
			}}
			currentModel={null}
			activeId={null}
			onSelect={handleSelect}
			onInfoClick={(modelName) => console.log('Info clicked:', modelName)}
		/>
	</div>
</Story>

<Story name="WithFavoritesOnly">
	<div class="w-80 rounded-lg border border-border bg-popover p-2 shadow-md">
		<ModelsSelectorList
			groups={{
				loaded: [],
				favorites: favoriteModels,
				available: []
			}}
			currentModel={null}
			activeId={null}
			onSelect={handleSelect}
			onInfoClick={(modelName) => console.log('Info clicked:', modelName)}
		/>
	</div>
</Story>

<Story name="ModelLoadedStates">
	<div class="w-80 rounded-lg border border-border bg-popover p-2 shadow-md">
		<div class="px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none">
			Server model states
		</div>
		<ModelsSelectorOption
			option={mockModel('model-idle', 'Model (idle)', 'meta')}
			isSelected={false}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('model-loading', 'Model (loading)', 'meta')}
			isSelected={false}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('model-loaded', 'Model (loaded)', 'meta')}
			isSelected={false}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('model-sleeping', 'Model (sleeping)', 'meta')}
			isSelected={false}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('model-failed', 'Model (failed)', 'meta')}
			isSelected={false}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
	</div>
</Story>

<Story name="ModelSelectedStates">
	<div class="w-80 rounded-lg border border-border bg-popover p-2 shadow-md">
		<div class="px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none">
			Selection states
		</div>
		<ModelsSelectorOption
			option={mockModel('normal-model', 'Normal Model', 'meta')}
			isSelected={false}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('selected-model', 'Selected Model', 'meta')}
			isSelected={true}
			isHighlighted={false}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('highlighted-model', 'Highlighted Model', 'meta')}
			isSelected={false}
			isHighlighted={true}
			isFav={false}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
		<ModelsSelectorOption
			option={mockModel('fav-model', 'Favorite Model', 'Qwen')}
			isSelected={false}
			isHighlighted={false}
			isFav={true}
			hideOrgName={true}
			onSelect={() => {}}
			onMouseEnter={() => {}}
			onKeyDown={() => {}}
		/>
	</div>
</Story>
