<script lang="ts">
	import { MODEL_VARIANT_BADGE_CLASS } from '$lib/constants';
	import type { ModelSidecarBadge } from '$lib/types/models';
	import { isAuxSidecar } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Draft sidecars available for the model. */
		draftSidecars?: ModelSidecarBadge[];
	}

	let { draftSidecars = [] }: Props = $props();

	let kinds = $derived(
		[...new SvelteSet(draftSidecars.map((badge) => badge.kind))].filter(
			(kind) => !isAuxSidecar(kind)
		)
	);
</script>

{#each kinds as kind (kind)}
	<span class={MODEL_VARIANT_BADGE_CLASS} title={`${kind.toUpperCase()} draft model available`}>
		{kind}
	</span>
{/each}
