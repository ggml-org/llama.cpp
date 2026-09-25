<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		/** Axis the region collapses along; a toolbar row collapses across. */
		axis?: 'height' | 'width';
		children: Snippet;
		open: boolean;
	}

	let { axis = 'height', children, open }: Props = $props();

	// rows stay mounted through the collapse transition so it can play
	const EXPAND_TRANSITION_MS = 200;
	// the initial state only: the effect below owns it afterwards
	// svelte-ignore state_referenced_locally
	let contentMounted = $state(open);

	$effect(() => {
		if (open) {
			contentMounted = true;

			return;
		}

		const timer = setTimeout(() => (contentMounted = false), EXPAND_TRANSITION_MS);

		return () => clearTimeout(timer);
	});
</script>

<div class="collapsible-region" data-axis={axis} data-expanded={open}>
	{#if contentMounted}
		<div class="collapsible-region-content">
			{@render children()}
		</div>
	{/if}
</div>

<style>
	/*
	 * The region animates between height 0 and height auto. `interpolate-size`
	 * lets the auto keyword take part in the interpolation, so the content needs
	 * no measured height to stay in sync with. Browsers without it fall back to
	 * an interpolating grid row, which snaps only in the oldest engines.
	 */
	.collapsible-region {
		/* clip, not hidden: hidden would make the region a scrollport, and the
		   sticky rows inside it would then never leave their opening position */
		overflow: clip;
		visibility: hidden;
		interpolate-size: allow-keywords;
		transition:
			height 200ms cubic-bezier(0.23, 1, 0.32, 1),
			width 200ms cubic-bezier(0.23, 1, 0.32, 1),
			visibility 200ms;
	}

	.collapsible-region[data-expanded='true'] {
		visibility: visible;
	}

	.collapsible-region[data-axis='height'] {
		height: 0;
	}

	.collapsible-region[data-axis='height'][data-expanded='true'] {
		height: auto;
	}

	.collapsible-region[data-axis='width'] {
		height: auto;
		width: 0;
	}

	.collapsible-region[data-axis='width'][data-expanded='true'] {
		width: auto;
	}

	@supports not (interpolate-size: allow-keywords) {
		.collapsible-region {
			display: grid;
		}

		.collapsible-region[data-axis='height'] {
			/* minmax(0, 1fr) pins the column to the region width: a plain auto
			   column would size to the content and push wide rows out of the list */
			grid-template-columns: minmax(0, 1fr);
			grid-template-rows: 0fr;
			/* the row owns the height here: leaving height: 0 in place would snap
			   the region shut before the row could interpolate */
			height: auto;
			transition:
				grid-template-rows 200ms cubic-bezier(0.23, 1, 0.32, 1),
				visibility 200ms;
		}

		.collapsible-region[data-axis='height'][data-expanded='true'] {
			grid-template-rows: 1fr;
		}

		.collapsible-region[data-axis='width'] {
			grid-template-columns: 0fr;
			grid-template-rows: minmax(0, 1fr);
			width: auto;
			transition:
				grid-template-columns 200ms cubic-bezier(0.23, 1, 0.32, 1),
				visibility 200ms;
		}

		.collapsible-region[data-axis='width'][data-expanded='true'] {
			grid-template-columns: 1fr;
		}

		.collapsible-region-content {
			min-height: 0;
			min-width: 0;
			overflow: clip;
		}
	}
</style>
