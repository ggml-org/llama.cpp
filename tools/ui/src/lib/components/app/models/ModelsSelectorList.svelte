<script lang="ts">
	import type { GroupedModelOptions, ModelItem } from './utils';
	import { ModelsSelectorOption } from '$lib/components/app';
	import { modelsStore } from '$lib/stores';

	interface Props {
		groups: GroupedModelOptions;
		currentModel: string | null;
		activeId: string | null;
		sectionHeaderClass?: string;
		orgHeaderClass?: string;
		onSelect: (modelId: string) => void;
		onInfoClick: (modelName: string) => void;
		renderOption?: import('svelte').Snippet<[ModelItem, boolean]>;
	}

	let {
		activeId,
		currentModel,
		groups,
		onInfoClick,
		onSelect,
		orgHeaderClass = 'px-2 py-2 text-[11px] font-semibold text-muted-foreground/50 select-none [&:not(:first-child)]:mt-1',
		renderOption,
		sectionHeaderClass = 'my-1 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none'
	}: Props = $props();
	let render = $derived(renderOption ?? defaultOption);
</script>

{#snippet defaultOption(item: ModelItem, hideOrgName: boolean)}
	{@const { option } = item}
	{@const isSelected = currentModel === option.model || activeId === option.id}
	{@const isFav = modelsStore.favoriteModelIds.has(option.model)}

	<ModelsSelectorOption
		{hideOrgName}
		{isFav}
		isHighlighted={false}
		{isSelected}
		{onInfoClick}
		onKeyDown={() => {}}
		onMouseEnter={() => {}}
		{onSelect}
		{option}
	/>
{/snippet}

{#if groups.loaded.length > 0}
	<p class={sectionHeaderClass}>Loaded models</p>

	{#each groups.loaded as item (`loaded-${item.option.id}`)}
		{@render render(item, false)}
	{/each}
{/if}

{#if groups.favorites.length > 0}
	<p class={sectionHeaderClass}>Favorite models</p>

	{#each groups.favorites as item (`fav-${item.option.id}`)}
		{@render render(item, true)}
	{/each}
{/if}

{#if groups.available.length > 0}
	<p class={sectionHeaderClass}>Available models</p>

	{#each groups.available as group (group.orgName)}
		{#if group.orgName}
			<p class={orgHeaderClass}>{group.orgName}</p>
		{/if}

		{#each group.items as item (item.option.id)}
			{@render render(item, true)}
		{/each}
	{/each}
{/if}
