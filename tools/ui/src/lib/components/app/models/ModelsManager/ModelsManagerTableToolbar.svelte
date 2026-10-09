<script lang="ts">
	import ModelsManagerFilters from './ModelsManagerFilters.svelte';
	import { hasActiveFilters } from './utils';
	import { X } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app/forms';
	import { Button } from '$lib/components/ui/button';
	import { type ModalityKey } from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import { backendsStore, deviceStore, uiStore } from '$lib/stores';
	import type { Snippet } from 'svelte';

	interface Props {
		/** Capabilities a model must have every one of. */
		capabilities?: ModelCapability[];
		/** Smallest context a model must support; 0 keeps every model. */
		contextLimit?: number;
		/** Keep only models that have a draft sidecar to speculate with. */
		draft?: boolean;
		filter?: string;
		/** Modalities a model must support at least one of. */
		modalities?: ModalityKey[];
		/** Repos each provider contributes to the current search, for the filter menu. */
		providerCounts?: Record<string, number>;
		/** Backend ids to keep; empty keeps every provider. */
		providers?: string[];
		/** Rendered at the toolbar's right end, past the filters. */
		toolbarEnd?: Snippet;
	}

	let {
		capabilities = $bindable<ModelCapability[]>([]),
		contextLimit = $bindable(0),
		draft = $bindable(false),
		filter = $bindable(''),
		modalities = $bindable<ModalityKey[]>([]),
		providerCounts = {},
		providers = $bindable<string[]>([]),
		toolbarEnd
	}: Props = $props();

	let hasFilters = $derived(
		hasActiveFilters(contextLimit, modalities, capabilities) || providers.length > 0 || draft
	);
	let filterInput = $state<HTMLInputElement | null>(null);

	// The search takes the focus the dialog would give its first control: the input
	// mounts a frame or two after the dialog opens (the toolbar renders below the
	// table in the DOM), so the poll retries until the ref lands or gives up.
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

<!-- Below md the search takes a row of its own and the filters follow it. -->
<div class="flex shrink-0 flex-wrap items-center gap-2 pb-4 max-md:gap-3 md:flex-nowrap">
	<SearchInput
		bind:ref={filterInput}
		bind:value={filter}
		class="w-full md:w-auto md:max-w-64"
		placeholder="Search your models"
		size={deviceStore.isMobile ? 'default' : 'sm'}
	/>

	<ModelsManagerFilters
		bind:capabilities
		bind:contextLimit
		bind:draft
		bind:modalities
		bind:providers
		backends={backendsStore.enabled}
		{providerCounts}
	/>

	{#if hasFilters}
		<Button
			class="gap-1.5 text-muted-foreground"
			onclick={() => {
				providers = [];
				contextLimit = 0;
				modalities = [];
				capabilities = [];
				draft = false;
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
