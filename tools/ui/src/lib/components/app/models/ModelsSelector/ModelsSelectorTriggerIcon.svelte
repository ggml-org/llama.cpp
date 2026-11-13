<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import { MODEL_ICON } from '$lib/constants';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		/** Selected model; its avatar decides the mark, the generic icon when absent. */
		option?: ModelOption | null;
	}

	let { class: className = 'h-3.5 w-3.5', option }: Props = $props();
</script>

{#if option}
	<ModelAvatar class={className} {option} showQuantBadge={false} showRepoOrgAvatar size={className}>
		{#snippet fallback()}
			<MODEL_ICON class={className} />
		{/snippet}
	</ModelAvatar>
{:else}
	<MODEL_ICON class={className} />
{/if}
