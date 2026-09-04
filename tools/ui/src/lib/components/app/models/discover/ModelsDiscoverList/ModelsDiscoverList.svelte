<script lang="ts">
	import ModelsDiscoverListItem from './ModelsDiscoverListItem.svelte';
	import ModelsDiscoverListItemSkeleton from './ModelsDiscoverListItemSkeleton.svelte';
	import type { HfModelInfo } from '$lib/types/huggingface';

	interface Props {
		models: HfModelInfo[];
		activeId?: string | null;
		loading?: boolean;
		loadingSkeletonRowsCount?: number;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let {
		activeId = null,
		loading = false,
		loadingSkeletonRowsCount = 8,
		models,
		onSelect,
		showBaseModelAvatar = false
	}: Props = $props();
</script>

<ul class="space-y-0.5 p-2">
	{#if loading}
		{#each Array(loadingSkeletonRowsCount) as _, index (index)}
			<ModelsDiscoverListItemSkeleton {index} />
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
