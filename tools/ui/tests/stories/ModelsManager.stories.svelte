<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsManagerModelsTable from '$lib/components/app/models/ModelsManager/ModelsManagerModelsTable.svelte';
	import type {
		ModelQuantGroup,
		ModelsTableGroup
	} from '$lib/components/app/models/ModelsManager/utils';
	import { MODELS_TABLE_GROUP_LABELS } from '$lib/constants';
	import { ModelGroupKind, ModelsTableGroupKind } from '$lib/enums';
	import { modelsStore } from '$lib/stores/models/index.svelte';
	import type { ModelOption } from '$lib/types/models';

	const { Story } = defineMeta({
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Components/ModelsManager'
	});

	const option = (model: string, name: string, contextLength?: number): ModelOption => ({
		capabilities: [],
		contextLength,
		id: model,
		model,
		name
	});

	const repo = (base: ModelOption, quants: ModelOption[]): ModelQuantGroup => ({
		base,
		key: base.model.split(':')[0],
		kind: ModelGroupKind.QUANTS,
		quants
	});

	const qwen = option('ggml-org/Qwen3.8-27B-GGUF:Q4_K_M', 'Qwen3.8-27B', 131_072);
	const qwenQ8 = option('ggml-org/Qwen3.8-27B-GGUF:Q8_0', 'Qwen3.8-27B', 131_072);
	const llama = option('unsloth/Llama-3.2-3B-GGUF:Q4_K_M', 'Llama-3.2-3B', 8192);
	const hiddenRepo = option('org/Experimental-MoE-GGUF:Q4_K_M', 'Experimental-MoE', 32_768);

	const section = (
		kind: ModelsTableGroupKind,
		items: ModelQuantGroup[],
		key = kind
	): ModelsTableGroup => ({
		items,
		key,
		kind,
		label: MODELS_TABLE_GROUP_LABELS[kind]
	});

	// the rows read the store for favorites, hidden ids and the load status
	modelsStore.models = [qwen, qwenQ8, llama, hiddenRepo];
	modelsStore.favoriteModelIds = new Set([llama.model]);
	modelsStore.hiddenModelIds = new Set([hiddenRepo.id]);
	modelsStore.routerModels = [
		{
			created: 0,
			id: qwen.model,
			in_cache: true,
			object: 'model',
			owned_by: 'llamacpp',
			path: `/models/${qwen.model}`,
			status: { value: 'loaded' }
		}
	];
</script>

<script lang="ts">
	let selectedId = $state<string | null>(null);

	const isFavorite = (picked: ModelOption) => modelsStore.favoriteModelIds.has(picked.model);
</script>

<Story name="Manager">
	<div class="h-[36rem] bg-background p-4">
		<ModelsManagerModelsTable
			groups={[
				section(ModelsTableGroupKind.LOADED, [repo(qwen, [qwen])]),
				section(ModelsTableGroupKind.FAVORITES, [repo(llama, [llama])]),
				section(ModelsTableGroupKind.LOCAL, [repo(qwen, [qwen, qwenQ8])]),
				section(ModelsTableGroupKind.HIDDEN, [repo(hiddenRepo, [hiddenRepo])])
			]}
			{isFavorite}
			onSelect={(picked: ModelOption) => (selectedId = picked.id)}
			overrides={{}}
			{selectedId}
		/>
	</div>
</Story>

<Story name="QuantsFolded">
	<div class="h-[24rem] bg-background p-4">
		<ModelsManagerModelsTable
			groups={[section(ModelsTableGroupKind.LOCAL, [repo(qwen, [qwen, qwenQ8])])]}
			{isFavorite}
			onSelect={(picked: ModelOption) => (selectedId = picked.id)}
			overrides={{}}
			{selectedId}
		/>
	</div>
</Story>

<Story name="Empty">
	<div class="h-[24rem] bg-background p-4">
		<ModelsManagerModelsTable
			groups={[]}
			{isFavorite}
			onSelect={(picked: ModelOption) => (selectedId = picked.id)}
			overrides={{}}
			{selectedId}
		/>
	</div>
</Story>
