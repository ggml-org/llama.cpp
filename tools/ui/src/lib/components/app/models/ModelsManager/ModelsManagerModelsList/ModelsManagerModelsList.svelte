<script lang="ts">
	import type { ModelsListFilter, ModelsListGroup } from '../utils';
	import ModelsManagerModelsListControls from './ModelsManagerModelsListControls.svelte';
	import ModelsManagerModelsListItem from './ModelsManagerModelsListItem.svelte';
	import { Cloud, Heart, Server } from '@lucide/svelte';
	import { CollapsibleSection } from '$lib/components/app';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		favorites: ModelOption[];
		filter?: string;
		groups: ModelsListGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onSelect: (option: ModelOption) => void;
		onToggleFavorite: (option: ModelOption) => void;
		selectedId: string | null;
		view?: ModelsListFilter;
	}

	let {
		favorites,
		filter = $bindable(''),
		groups,
		isFavorite,
		onSelect,
		onToggleFavorite,
		selectedId,
		view = $bindable<ModelsListFilter>('all')
	}: Props = $props();

	const headerClass = 'flex w-full cursor-pointer items-center gap-1.5 px-2 py-1.5 text-left';
</script>

<div class="flex h-full min-h-0 flex-col gap-3">
	<ModelsManagerModelsListControls bind:filter bind:view />

	<div class="min-h-0 flex-1 space-y-1 overflow-y-auto">
		{#if favorites.length > 0}
			<CollapsibleSection triggerClass={headerClass}>
				{#snippet trigger()}
					<Heart class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

					<span class="text-xs font-semibold text-muted-foreground">Favorites</span>

					<span class="text-xs text-muted-foreground/70">{favorites.length}</span>
				{/snippet}

				{#each favorites as option (option.id)}
					<ModelsManagerModelsListItem
						isFavorite={isFavorite(option)}
						isSelected={selectedId === option.id}
						onSelect={() => onSelect(option)}
						onToggleFavorite={() => onToggleFavorite(option)}
						{option}
					/>
				{/each}
			</CollapsibleSection>
		{/if}

		{#each groups as group (group.backendId)}
			<CollapsibleSection triggerClass={headerClass}>
				{#snippet trigger()}
					{#if group.isLocal}
						<Server class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
					{:else}
						<Cloud class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
					{/if}

					<span class="text-xs font-semibold text-muted-foreground">{group.label}</span>

					<span class="text-xs text-muted-foreground/70">{group.items.length}</span>
				{/snippet}

				{#each group.items as option (option.id)}
					<ModelsManagerModelsListItem
						isFavorite={isFavorite(option)}
						isSelected={selectedId === option.id}
						onSelect={() => onSelect(option)}
						onToggleFavorite={() => onToggleFavorite(option)}
						{option}
					/>
				{/each}
			</CollapsibleSection>
		{/each}

		{#if groups.length === 0 && favorites.length === 0}
			<p class="px-2 py-6 text-center text-sm text-muted-foreground">No models found.</p>
		{/if}
	</div>
</div>
