<script lang="ts">
	import { Check, ChevronDown } from '@lucide/svelte';
	import { DropdownMenuSearchable, ModelsSelectorList } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { modelsStore } from '$lib/stores';
	import { filterModelOptions, groupFavoriteOptions, groupModelOptions } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		id: string;
		onSelect: (model: string) => void;
		placeholder: string;
		value: string;
		/** Entry above the model list that stores its own value, e.g. an automatic choice. */
		emptyOption?: { label: string; value: string };
		/** Restricts the model list, e.g. to audio-capable models. */
		filter?: (model: ModelOption) => boolean;
	}

	let { emptyOption, filter, id, onSelect, placeholder, value }: Props = $props();

	let open = $state(false);
	let searchTerm = $state('');

	let models = $derived(filter ? modelsStore.models.filter(filter) : modelsStore.models);
	let filteredOptions = $derived(filterModelOptions(models, searchTerm));
	let loaded = $derived(
		filteredOptions
			.filter((option) => modelsStore.isModelLoaded(option.model))
			.map((option) => ({ option }))
	);
	let loadedIds = $derived(new SvelteSet(loaded.map((item) => item.option.id)));
	let favorites = $derived(
		groupFavoriteOptions(
			filteredOptions.filter((option) => !loadedIds.has(option.id)),
			modelsStore.favoriteModelIds
		)
	);
	let groups = $derived(
		groupModelOptions(
			filteredOptions.filter(
				(option) => !loadedIds.has(option.id) && !modelsStore.favoriteModelIds.has(option.model)
			)
		)
	);
	// The stored value survives a model leaving the router list, so the raw
	// id is shown when no option matches it.
	let triggerLabel = $derived.by(() => {
		if (emptyOption && value === emptyOption.value) return emptyOption.label;

		return modelsStore.models.find((m) => m.model === value)?.name || value || placeholder;
	});

	function handleOpenChange(next: boolean) {
		open = next;
		searchTerm = '';
	}

	function handleSelect(modelId: string) {
		const option = modelsStore.models.find((m) => m.id === modelId);

		if (!option) return;

		onSelect(option.model);
		handleOpenChange(false);
	}
</script>

<DropdownMenu.Root onOpenChange={handleOpenChange} {open}>
	<DropdownMenu.Trigger {id}>
		{#snippet child({ props })}
			<Button {...props} class="w-full justify-between font-normal" variant="outline">
				<span class="truncate">{triggerLabel}</span>

				<ChevronDown class="h-4 w-4 shrink-0 opacity-50" />
			</Button>
		{/snippet}
	</DropdownMenu.Trigger>

	<DropdownMenu.Content
		class="w-100 max-w-[calc(100vw-2rem)] pt-0"
		onOpenAutoFocus={(event) => event.preventDefault()}
	>
		<DropdownMenuSearchable
			bind:searchValue={searchTerm}
			emptyMessage="No models found."
			isEmpty={filteredOptions.length === 0 && !emptyOption}
			placeholder="Search models..."
		>
			<div class="max-h-72 overflow-y-auto">
				{#if emptyOption}
					<DropdownMenu.Item
						class="flex items-center justify-between"
						onclick={() => onSelect(emptyOption.value)}
					>
						<span class="truncate">{emptyOption.label}</span>

						{#if value === emptyOption.value}
							<Check class="h-4 w-4 shrink-0" />
						{/if}
					</DropdownMenu.Item>
				{/if}

				<ModelsSelectorList
					activeId={null}
					currentModel={value}
					{favorites}
					{groups}
					{loaded}
					onSelect={handleSelect}
				/>
			</div>
		</DropdownMenuSearchable>
	</DropdownMenu.Content>
</DropdownMenu.Root>
