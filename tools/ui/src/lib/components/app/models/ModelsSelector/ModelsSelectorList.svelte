<script lang="ts">
	import ModelsSelectorDownloadItem from './ModelsSelectorDownloadItem.svelte';
	import { Heart, Power } from '@lucide/svelte';
	import { GroupedList, ModelAvatar, ModelsSelectorOption } from '$lib/components/app';
	import { ModelsSection } from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import Logo from '$lib/components/app/misc/Logo.svelte';
	import {
		type GroupedModelOptions,
		type ModelItem,
		windowLocalGroups
	} from '$lib/components/app/navigation/utils';
	import { LOCAL_BACKEND_ID, MODEL_ROW_WINDOW } from '$lib/constants';
	import { ModelDownloadConfirmAction } from '$lib/enums';
	import { modelsStore, settingsStore } from '$lib/stores';
	import { groupModelFamilies, type ModelFamilyGroup } from '$lib/utils/model-families';

	interface Props {
		groups: GroupedModelOptions;
		currentModel: string | null;
		activeId: string | null;
		sectionHeaderClass?: string;
		onSelect: (modelId: string) => void;
		renderOption?: import('svelte').Snippet<[ModelItem, boolean]>;
		/** Favorite models of every backend, listed in their own section. */
		favorites?: ModelItem[];
		/** Loaded models of every llama-compat backend, leading the list. */
		loaded?: ModelItem[];
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
		loaded = [],
		onProviderBack,
		onProviderOpen,
		onSelect,
		renderOption,
		sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none',
		showOrgName = true
	}: Props = $props();
	let render = $derived(renderOption ?? defaultOption);
	// A large local catalog is mounted a window at a time; the sentinel at the end
	// of the list grows the window when it scrolls into view.
	let visibleCount = $state(MODEL_ROW_WINDOW);
	let sentinelEl = $state<HTMLElement | null>(null);
	const localGroups = $derived(windowLocalGroups(groups, visibleCount));
	const localRowCount = $derived(
		groups.loaded.length + groups.available.reduce((count, group) => count + group.items.length, 0)
	);
	const hasMoreLocal = $derived(localGroups.shown < localRowCount);

	$effect(() => {
		const sentinel = sentinelEl;

		if (!sentinel || !hasMoreLocal) return;

		// scroll does not bubble, so listen in the capture phase and ask the
		// sentinel where it is instead of guessing which ancestor scrolls
		const onScroll = () => {
			const rect = sentinel.getBoundingClientRect();

			// grow only when the end of the list is actually on screen, otherwise
			// every scroll event of the page would mount the whole catalog
			if (rect.top > window.innerHeight || rect.bottom < 0) return;

			visibleCount += MODEL_ROW_WINDOW;
		};

		document.addEventListener('scroll', onScroll, { capture: true, passive: true });
		onScroll();

		return () => document.removeEventListener('scroll', onScroll, { capture: true });
	});
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

{#snippet familyHeading({ group: family }: { group: ModelFamilyGroup<ModelItem> })}
	<div class="flex items-center gap-2 px-2 py-1.5 select-none">
		<ModelAvatar
			option={family.entries[0].option}
			showBaseModelAvatar
			showQuantBadge={false}
			size="size-6"
		/>

		<span class="truncate text-[13px] font-semibold text-muted-foreground/70">
			{family.label}
		</span>

		<span class="text-xs text-muted-foreground">
			{family.entries.length} model{family.entries.length === 1 ? '' : 's'}
		</span>
	</div>
{/snippet}

{#snippet listItem({ depth, entry }: { depth: number; entry: ModelItem })}
	<div style="padding-left: {depth * 16}px">{@render render(entry, !showOrgName)}</div>
{/snippet}

{#snippet listRows(items: ModelItem[], prefix: string)}
	{#if settingsStore.config.groupModelsByFamily}
		<GroupedList
			group={familyHeading}
			groupStateKey={prefix}
			groups={groupModelFamilies(items, (row) => row.option.model).map((family) => ({
				entries: family.entries,
				group: family,
				key: `${prefix}-${family.key}`
			}))}
			item={listItem}
			keyOf={(row) => `${prefix}-${row.option.id}`}
			stickyStyle="top: calc(var(--dropdown-sticky-height, 0px) + 2.25rem - 1px)"
		/>
	{:else}
		<GroupedList item={listItem} {items} keyOf={(row) => `${prefix}-${row.option.id}`} />
	{/if}
{/snippet}

{#snippet defaultOption(item: ModelItem, hideOrgName: boolean)}
	{@const { option } = item}
	{@const isSelected = currentModel === option.model || activeId === option.id}
	{@const isFav = modelsStore.favoriteModelIds.has(option.model)}

	<ModelsSelectorOption
		{hideOrgName}
		{isFav}
		isHighlighted={false}
		{isSelected}
		onKeyDown={() => {}}
		onMouseEnter={() => {}}
		{onSelect}
		{option}
		showBaseModelAvatar={!settingsStore.config.groupModelsByFamily}
		showRepoOrgAvatar={settingsStore.config.groupModelsByFamily}
	/>
{/snippet}

{#if loaded.length > 0}
	<ModelsSection
		count={loaded.length}
		label="Loaded models"
		persistKey="loaded"
		revealChevronOnHover
		sticky
	>
		{#snippet icon()}
			<Power class="h-3.5 w-3.5 shrink-0" />
		{/snippet}

		{#each loaded as item (item.option.id)}
			{@render render(item, !showOrgName)}
		{/each}
	</ModelsSection>
{/if}

{#if favorites.length > 0}
	<!-- Favorites come first; the sections below skip them -->
	<ModelsSection label="Favorites" persistKey="favorites" revealChevronOnHover sticky>
		{#snippet icon()}
			<Heart class="h-3.5 w-3.5 shrink-0" />
		{/snippet}

		{#each favorites as item (`fav-${item.option.id}`)}
			{@render render(item, !showOrgName)}
		{/each}
	</ModelsSection>
{/if}

{#if getDownloadEntries.length > 0}
	<p class={headerClass} style={headerStyle}>Download in progress</p>

	{#each getDownloadEntries as entry (entry.repoWithTag)}
		<ModelsSelectorDownloadItem {entry} onRequestCancel={requestCancel} {showOrgName} />
	{/each}
{/if}

{#snippet localRows()}
	<!-- the loaded models first -->
	{#each localGroups.loaded as item (`loaded-${item.option.id}`)}
		{@render render(item, !showOrgName)}
	{/each}

	{@render listRows(
		localGroups.available.flatMap((group) => group.items),
		'local'
	)}

	{#if hasMoreLocal}
		<div bind:this={sentinelEl} aria-hidden="true" class="h-px"></div>
	{/if}
{/snippet}

{#if groups.loaded.length > 0 || groups.available.length > 0}
	<ModelsSection label="Local models" persistKey={LOCAL_BACKEND_ID} revealChevronOnHover sticky>
		{#snippet icon()}
			<Logo class="shrink-0" style="--size: 0.875rem" />
		{/snippet}

		{@render localRows()}
	</ModelsSection>
{/if}

<!-- One section per remote provider. -->
{#each groups.providers as provider (provider.backendId)}
	<ModelsSection
		backendId={provider.backendId}
		error={Boolean(provider.error)}
		label={provider.name}
		loading={provider.loading}
		onBack={onProviderBack}
		persistKey={provider.backendId}
		revealChevronOnHover
		sticky
	>
		{#if provider.items.length > 0}
			{@render listRows(provider.items, provider.backendId)}

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
	</ModelsSection>
{/each}

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.CANCEL}
	onClose={() => (cancelOpen = false)}
	open={cancelOpen}
	repoWithTag={pendingCancel}
/>
