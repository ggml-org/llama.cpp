<script lang="ts">
	import { BackendIcon } from '$lib/components/app/backends';
	import { Logo } from '$lib/components/app/misc';
	import { LOCAL_BACKEND_ID, MODEL_SELECTOR_ICON } from '$lib/constants';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';

	interface Props {
		class?: string;
		/** Selected model; its provider decides the mark, the generic icon when absent. */
		option?: ModelOption | null;
	}

	let { class: className = 'size-1', option }: Props = $props();

	// the bundled server has no favicon to resolve, its mark is the llama.cpp logo
	let isLocal = $derived(getBackend(option?.backendId)?.id === LOCAL_BACKEND_ID);
</script>

{#if option}
	<BackendIcon backend={getBackend(option.backendId)} class={className}>
		{#snippet fallback()}
			{#if isLocal}
				<Logo class={className} style="--size: 95%; margin-top: 1px;" />
			{:else}
				<MODEL_SELECTOR_ICON class={className} />
			{/if}
		{/snippet}
	</BackendIcon>
{:else}
	<MODEL_SELECTOR_ICON class={className} />
{/if}
