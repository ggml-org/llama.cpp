<script lang="ts">
	import ModelsDiscoverListItem from './ModelsDiscoverListItem.svelte';
	import ModelsDiscoverListItemSkeleton from './ModelsDiscoverListItemSkeleton.svelte';
	import type { HfModelInfo } from '$lib/types/huggingface';

	interface Props {
		models: HfModelInfo[];
		activeId?: string | null;
		/** Render skeleton rows instead of the list while models load. */
		loading?: boolean;
		/** Number of skeleton rows to render while loading. */
		loadingCount?: number;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let {
		activeId = null,
		loading = false,
		loadingCount = 8,
		models,
		onSelect,
		showBaseModelAvatar = false
	}: Props = $props();
</script>

<ul class="space-y-0.5 p-2">
	{#if loading}
		{#each Array(loadingCount) as _, index (index)}
			<ModelsDiscoverListItemSkeleton />
		{/each}
	{:else}
		{#each models as model (model.id)}
			<ModelsDiscoverListItem
				active={model.id === activeId}
				{model}
				{onSelect}
				{showBaseModelAvatar}
			/>
		{/each}
	{/if}
</ul>
