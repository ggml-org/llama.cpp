<script lang="ts">
	import ModelsSelectorDownloadItem from './ModelsSelectorDownloadItem.svelte';
	import { ChevronLeft, CircleAlert, Heart, Loader2 } from '@lucide/svelte';
	import { ModelsSelectorOption } from '$lib/components/app';
	import { CollapsibleSection } from '$lib/components/app';
	import { BackendIcon } from '$lib/components/app/backends';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import Logo from '$lib/components/app/misc/Logo.svelte';
	import type { GroupedModelOptions, ModelItem } from '$lib/components/app/navigation/utils';
	import { ModelDownloadConfirmAction } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import { getBackend } from '$lib/utils/api-base';

	interface Props {
		groups: GroupedModelOptions;
		currentModel: string | null;
		activeId: string | null;
		sectionHeaderClass?: string;
		onSelect: (modelId: string) => void;
		onInfoClick: (modelName: string) => void;
		renderOption?: import('svelte').Snippet<[ModelItem, boolean]>;
		/** Favorite models of every backend, listed in their own section. */
		favorites?: ModelItem[];
		/** Show the organization name in every model id of the list. */
		showOrgName?: boolean;
		/** Open one provider's full list, offered when a section is cut short. */
		onProviderOpen?: (backendId: string) => void;
		/** Leave the drilled-in provider; enables the back affordance. */
		onProviderBack?: () => void;
	}

	let {
		activeId,
		currentModel,
		favorites = [],
		groups,
		onInfoClick,
		onProviderBack,
		onProviderOpen,
		onSelect,
		renderOption,
		sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none',
		showOrgName = true
	}: Props = $props();
	let render = $derived(renderOption ?? defaultOption);
	// section headers stick right below the search/tabs block of the dropdown
	// scrollport; `--dropdown-sticky-height` is set by DropdownMenuSearchable
	// and falls back to 0 in surfaces without one (the mobile sheet)
	let headerClass = $derived(`${sectionHeaderClass} sticky z-10 bg-popover`);
	const headerStyle = 'top: var(--dropdown-sticky-height, 0px)';

	/** In-flight / paused downloads, tracked by the status feed. */
	let getDownloadEntries = $derived(modelsStore.status.getDownloadEntries());

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
		showBaseModelAvatar
	/>
{/snippet}

{#if favorites.length > 0}
	<!-- Favorites come first; the sections below skip them -->
	<CollapsibleSection
		revealChevronOnHover
		triggerClass="{headerClass} flex w-full cursor-pointer items-center gap-1.5 text-left"
		triggerStyle={headerStyle}
	>
		{#snippet trigger()}
			<Heart class="h-3.5 w-3.5 shrink-0" />

			Favorites
		{/snippet}

		{#each favorites as item (`fav-${item.option.id}`)}
			{@render render(item, !showOrgName)}
		{/each}
	</CollapsibleSection>
{/if}

{#if getDownloadEntries.length > 0}
	<p class={headerClass} style={headerStyle}>Download in progress</p>

	{#each getDownloadEntries as entry (entry.repoWithTag)}
		<ModelsSelectorDownloadItem {entry} onRequestCancel={requestCancel} {showOrgName} />
	{/each}
{/if}

{#if groups.loaded.length > 0 || groups.available.length > 0}
	<!-- Local models: one list, the loaded ones first. -->
	<CollapsibleSection
		revealChevronOnHover
		triggerClass="{headerClass} flex w-full cursor-pointer items-center gap-1.5 text-left"
		triggerStyle={headerStyle}
	>
		{#snippet trigger()}
			<Logo class="shrink-0" style="--size: 0.875rem" />

			Local models
		{/snippet}

		{#each groups.loaded as item (`loaded-${item.option.id}`)}
			{@render render(item, !showOrgName)}
		{/each}

		{#each groups.available as group (group.orgName)}
			{#each group.items as item (item.option.id)}
				{@render render(item, !showOrgName)}
			{/each}
		{/each}
	</CollapsibleSection>
{/if}

<!-- One section per remote provider. -->
{#each groups.providers as provider (provider.backendId)}
	<CollapsibleSection
		revealChevronOnHover
		triggerClass="{headerClass} flex w-full cursor-pointer items-center gap-1.5 text-left"
		triggerStyle={headerStyle}
	>
		{#snippet trigger()}
			{#if onProviderBack}
				<button
					aria-label="Back to all providers"
					class="-ml-1 inline-flex shrink-0 cursor-pointer items-center rounded-sm p-0.5 text-muted-foreground transition hover:bg-muted/60 hover:text-foreground"
					onclick={onProviderBack}
					type="button"
				>
					<ChevronLeft class="h-3.5 w-3.5" />
				</button>
			{/if}

			<BackendIcon backend={getBackend(provider.backendId)} class="h-3.5 w-3.5" />

			{provider.name}

			{#if provider.loading}
				<Loader2 class="h-3 w-3 animate-spin" />
			{:else if provider.error}
				<CircleAlert class="h-3 w-3 text-destructive" />
			{/if}
		{/snippet}

		{#if provider.items.length > 0}
			{#each provider.items as item (`${provider.backendId}-${item.option.id}`)}
				{@render render(item, !showOrgName)}
			{/each}

			{#if onProviderOpen && provider.matched > provider.items.length}
				<!-- same box as a model row, it opens the provider's full list -->
				<button
					class="flex w-full cursor-pointer items-center gap-2 rounded-sm p-2 text-left text-sm text-muted-foreground transition hover:bg-accent hover:text-foreground focus:outline-none"
					onclick={() => onProviderOpen(provider.backendId)}
					type="button"
				>
					+ {provider.matched - provider.items.length} more
				</button>
			{/if}
		{:else if provider.catalog === 0}
			<p class="px-4 pb-2 text-xs text-muted-foreground">
				{provider.error ?? (provider.loading ? 'Loading models...' : 'No models')}
			</p>
		{/if}
	</CollapsibleSection>
{/each}

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.CANCEL}
	onClose={() => (cancelOpen = false)}
	open={cancelOpen}
	repoWithTag={pendingCancel}
/>
