<script lang="ts">
	import { MODEL_VARIANT_BADGE_CLASS } from '$lib/constants';
	import type { ModelSidecarBadge } from '$lib/types/models';
	import { isAuxSidecar } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		/** Draft sidecars available for the model, each with its own quant. */
		draftSidecars?: ModelSidecarBadge[];
	}

	let { draftSidecars = [] }: Props = $props();

	// the router listing and the --model-draft args both report a model's drafts, so
	// the same sidecar can arrive twice
	let badges = $derived.by(() => {
		const seen = new SvelteSet<string>();

		return draftSidecars.filter((badge) => {
			if (isAuxSidecar(badge.kind)) return false;

			const key = `${badge.kind}:${badge.quant ?? ''}`;

			if (seen.has(key)) return false;

			seen.add(key);

			return true;
		});
	});
</script>

{#each badges as badge (badge.kind + (badge.quant ?? ''))}
	<span aria-hidden="true" class="text-muted-foreground">+</span>

	<span
		class={MODEL_VARIANT_BADGE_CLASS}
		title="{badge.kind.toUpperCase()} draft model{badge.quant ? ` at ${badge.quant}` : ''}"
	>
		{badge.kind}
	</span>
{/each}
