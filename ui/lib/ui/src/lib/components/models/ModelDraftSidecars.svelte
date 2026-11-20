<!--
	A model's draft sidecars, one badge each, titled with the sidecar's own quant when it has one.
-->
<script lang="ts">
	import { MODEL_VARIANT_BADGE_CLASS } from '../../models/constants';
	import type { ModelDraftSidecarBadge } from '../../models/types';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Draft sidecars available for the model, each with its own quant. */
		draftSidecars?: ModelDraftSidecarBadge[];
	}

	let { draftSidecars = [] }: Props = $props();

	// an app can learn of the same sidecar from more than one place
	let badges = $derived.by(() => {
		const seen = new SvelteSet<string>();

		return draftSidecars.filter((badge) => {
			const key = `${badge.kind}:${badge.quant ?? ''}`;

			if (seen.has(key)) return false;

			seen.add(key);

			return true;
		});
	});
</script>

{#each badges as badge (badge.kind + (badge.quant ?? ''))}
	<span
		class={MODEL_VARIANT_BADGE_CLASS}
		title="{badge.kind.toUpperCase()} draft model{badge.quant ? ` at ${badge.quant}` : ''}"
	>
		{badge.kind}
	</span>
{/each}
