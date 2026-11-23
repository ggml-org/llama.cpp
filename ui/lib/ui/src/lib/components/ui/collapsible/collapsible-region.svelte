<script lang="ts">
	import { cn } from '../utils';
	import { COLLAPSIBLE_TRANSITION_MS } from '../../../constants';
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	interface Props extends HTMLAttributes<HTMLDivElement> {
		children?: Snippet;
		open: boolean;
	}

	let { children, class: className, open, ...restProps }: Props = $props();

	// The body stays mounted through the collapse so it can play, and unmounts once
	// it is over: a closed collapsible costs nothing, even while its content would
	// re-render on every streamed token. This is only the initial state, the effect
	// below owns it afterwards.
	// svelte-ignore state_referenced_locally
	let contentMounted = $state(open);

	// bits-ui rebuilds the snippet args (with `open`) when its transition ends; the derived keeps that from restarting the unmount timer
	const isOpen = $derived(open);

	$effect(() => {
		if (isOpen) {
			contentMounted = true;

			return;
		}

		const timer = setTimeout(() => (contentMounted = false), COLLAPSIBLE_TRANSITION_MS);

		return () => clearTimeout(timer);
	});
</script>

<div {...restProps} class={cn('collapsible-region', className)} data-expanded={isOpen}>
	{#if contentMounted}
		<div class="collapsible-region-content">
			{@render children?.()}
		</div>
	{/if}
</div>

<style>
	/* `interpolate-size` lets `height: auto` take part in the transition, so the content
	   needs no measured height. Older engines fall back to an interpolating grid row. */
	.collapsible-region {
		height: 0;
		/* clip, not hidden: hidden would make the region a scrollport, and the
		   sticky rows inside it would then never leave their opening position */
		overflow: clip;
		visibility: hidden;
		interpolate-size: allow-keywords;
		transition:
			height 200ms cubic-bezier(0.23, 1, 0.32, 1),
			visibility 200ms;
	}

	.collapsible-region[data-expanded='true'] {
		height: auto;
		visibility: visible;
	}

	@supports not (interpolate-size: allow-keywords) {
		.collapsible-region {
			display: grid;
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

		.collapsible-region[data-expanded='true'] {
			grid-template-rows: 1fr;
		}

		.collapsible-region-content {
			min-height: 0;
			min-width: 0;
			overflow: clip;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.collapsible-region {
			transition: none;
		}
	}
</style>
