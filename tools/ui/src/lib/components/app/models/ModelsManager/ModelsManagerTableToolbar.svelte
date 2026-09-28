<script lang="ts">
	import ModelsManagerFilters from './ModelsManagerFilters.svelte';
	import { type ModalityKey } from './utils';
	import { X } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app/forms';
	import { Button } from '$lib/components/ui/button';
	import { ModelCapability } from '$lib/enums';
	import { uiStore } from '$lib/stores';
	import type { Snippet } from 'svelte';

	interface Props {
		/** Capabilities a model must have every one of. */
		capabilities?: ModelCapability[];
		/** Smallest context a model must support; 0 keeps every model. */
		contextLimit?: number;
		filter?: string;
		/** Modalities a model must support at least one of. */
		modalities?: ModalityKey[];
		/** Rendered at the toolbar's right end, past the filters. */
		toolbarEnd?: Snippet;
	}

	let {
		capabilities = $bindable<ModelCapability[]>([]),
		contextLimit = $bindable(0),
		filter = $bindable(''),
		modalities = $bindable<ModalityKey[]>([]),
		toolbarEnd
	}: Props = $props();

	let hasFilters = $derived(contextLimit > 0 || modalities.length > 0 || capabilities.length > 0);
	let filterInput = $state<HTMLInputElement | null>(null);

	// the dialog hands focus to its first control, so the filter takes it instead
	$effect(() => {
		if (!uiStore.manageModelsOpen) return;

		let frames = 0;
		let handle = requestAnimationFrame(function focusFilter() {
			if (filterInput) {
				filterInput.focus({ preventScroll: true });

				return;
			}

			if (frames++ < 20) handle = requestAnimationFrame(focusFilter);
		});

		return () => cancelAnimationFrame(handle);
	});
</script>

<div class="flex shrink-0 items-center gap-2 pb-4">
	<SearchInput
		bind:ref={filterInput}
		bind:value={filter}
		class="max-w-64"
		placeholder="Search your models"
		size="sm"
	/>

	<ModelsManagerFilters bind:capabilities bind:contextLimit bind:modalities />

	{#if hasFilters}
		<Button
			class="gap-1.5 text-muted-foreground"
			onclick={() => {
				contextLimit = 0;
				modalities = [];
				capabilities = [];
			}}
			size="sm"
			variant="ghost"
		>
			<X class="h-3.5 w-3.5" />

			Clear filters
		</Button>
	{/if}

	<div class="ml-auto flex items-center gap-2">
		{@render toolbarEnd?.()}
	</div>
</div>
