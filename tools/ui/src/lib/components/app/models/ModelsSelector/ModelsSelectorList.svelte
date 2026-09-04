<script lang="ts">
	import ModelsSelectorDownloadItem from './ModelsSelectorDownloadItem.svelte';
	import { ModelsSelectorOption } from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import type { GroupedModelOptions, ModelItem } from '$lib/components/app/navigation/utils';
	import { DownloadConfirmAction } from '$lib/enums';
	import { modelsStore } from '$lib/stores';

	interface Props {
		groups: GroupedModelOptions;
		currentModel: string | null;
		activeId: string | null;
		sectionHeaderClass?: string;
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
		renderOption,
		sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none'
	}: Props = $props();
	let render = $derived(renderOption ?? defaultOption);

	/** In-flight / paused downloads, tracked by the status feed. */
	let downloadEntries = $derived(modelsStore.status.downloadEntries());

	// Cancel is confirmed once for the whole list rather than per download row, so
	// a single dialog instance is mounted however many downloads are in flight.
	// The target is kept while the dialog closes so its copy stays rendered.
	let pendingCancel = $state('');
	let cancelOpen = $state(false);

	function requestCancel(repoWithTag: string) {
		pendingCancel = repoWithTag;
		cancelOpen = true;
	}
</script>

{#snippet defaultOption(item: ModelItem, _hideOrgName: boolean)}
	{@const { option } = item}
	{@const isSelected = currentModel === option.model || activeId === option.id}
	{@const isFav = modelsStore.favoriteModelIds.has(option.model)}

	<ModelsSelectorOption
		hideOrgName
		{isFav}
		isHighlighted={false}
		{isSelected}
		{onInfoClick}
		onKeyDown={() => {}}
		onMouseEnter={() => {}}
		{onSelect}
		{option}
		showBaseModelAvatar
	/>
{/snippet}

{#if groups.favorites.length > 0}
	<p class={sectionHeaderClass}>Favorite models</p>

	{#each groups.favorites as item (`fav-${item.option.id}`)}
		{@render render(item, true)}
	{/each}
{/if}

{#if downloadEntries.length > 0}
	<p class={sectionHeaderClass}>Download in progress</p>

	{#each downloadEntries as entry (entry.repoWithTag)}
		<ModelsSelectorDownloadItem {entry} onRequestCancel={requestCancel} />
	{/each}
{/if}

{#if groups.loaded.length > 0}
	<p class={sectionHeaderClass}>Loaded models</p>

	{#each groups.loaded as item (`loaded-${item.option.id}`)}
		{@render render(item, false)}
	{/each}
{/if}

{#if groups.available.length > 0}
	<h2 class={sectionHeaderClass}>Downloaded models</h2>

	{#each groups.available as group (group.orgName)}
		{#each group.items as item (item.option.id)}
			{@render render(item, true)}
		{/each}
	{/each}
{/if}

<DialogConfirmDownload
	action={DownloadConfirmAction.CANCEL}
	onClose={() => (cancelOpen = false)}
	open={cancelOpen}
	repoWithTag={pendingCancel}
/>
