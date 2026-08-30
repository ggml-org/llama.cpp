<script lang="ts" module>
	import { mockGemma, mockListModels, mockQwen } from './fixtures/models-discover';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ModelsDiscoverAvatar from '$lib/components/app/models/discover/ModelsDiscoverAvatar.svelte';
	import ModelsDiscoverInfo from '$lib/components/app/models/discover/ModelsDiscoverInfo.svelte';
	import ModelsDiscoverItem from '$lib/components/app/models/discover/ModelsDiscoverItem.svelte';
	import ModelsDiscoverList from '$lib/components/app/models/discover/ModelsDiscoverList.svelte';

	const { Story } = defineMeta({
		tags: ['autodocs'],
		title: 'Models/Discover/List'
	});
</script>

<Story name="Avatar">
	<div class="flex flex-col gap-4 p-4">
		<div class="flex items-center gap-3">
			<ModelsDiscoverAvatar org="ggml-org" />

			<span class="text-sm">repo org only</span>
		</div>

		<div class="flex items-center gap-3">
			<ModelsDiscoverAvatar org="ggml-org" quantOrg="Qwen" />

			<span class="text-sm">base org + quant corner badge</span>
		</div>
	</div>
</Story>

<Story name="Info">
	<div class="flex w-80 flex-col gap-2 p-4">
		<ModelsDiscoverInfo model={mockGemma} />

		<ModelsDiscoverInfo model={mockQwen} />
	</div>
</Story>

<Story name="Item">
	<ul class="w-96 p-2">
		<ModelsDiscoverItem active model={mockGemma} />

		<ModelsDiscoverItem model={mockQwen} />
	</ul>
</Story>

<Story name="List">
	<div class="h-64 w-96 overflow-y-auto">
		<ModelsDiscoverList activeId={mockGemma.id} models={mockListModels} />
	</div>
</Story>
