<script lang="ts">
	import type { ModelsListFilter, ModelsListGroup } from '../utils';
	import ModelsManagerModelsListControls from './ModelsManagerModelsListControls.svelte';
	import ModelsManagerModelsListItem from './ModelsManagerModelsListItem.svelte';
	import { Heart } from '@lucide/svelte';
	import { Logo, ModelsSection } from '$lib/components/app';
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
</script>

<div class="flex h-full min-h-0 flex-col gap-3">
	<ModelsManagerModelsListControls bind:filter bind:view />

	<div class="min-h-0 flex-1 space-y-1 overflow-y-auto">
		{#if favorites.length > 0}
			<ModelsSection count={favorites.length} label="Favorites" sticky>
				{#snippet icon()}
					<Heart class="h-3.5 w-3.5 shrink-0" />
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
			</ModelsSection>
		{/if}

		{#each groups as group (group.backendId)}
			<ModelsSection
				backendId={group.isLocal ? undefined : group.backendId}
				count={group.items.length}
				label={group.label}
				sticky
			>
				{#snippet icon()}
					{#if group.isLocal}
						<Logo class="shrink-0" style="--size: 0.875rem" />
					{/if}
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
			</ModelsSection>
		{/each}

		{#if groups.length === 0 && favorites.length === 0}
			<p class="px-2 py-6 text-center text-sm text-muted-foreground">No models found.</p>
		{/if}
	</div>
</div>
