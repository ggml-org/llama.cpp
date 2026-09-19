<script lang="ts">
	import { ChevronDown, ChevronUp } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import type { Snippet } from 'svelte';

	interface Props {
		/** Collapsible content. */
		children: Snippet;
		/** Trigger content; this component appends the chevron. */
		trigger: Snippet;
		/** Render the trigger as a dropdown menu item, so menu keyboard navigation reaches it. */
		inMenu?: boolean;
		/** Start expanded. */
		open?: boolean;
		/** Hide the chevron while expanded, until the trigger is hovered. */
		revealChevronOnHover?: boolean;
		/** Where the trigger sits relative to the content. */
		triggerPosition?: 'bottom' | 'top';
		triggerClass?: string;
		triggerStyle?: string;
	}

	let {
		children,
		inMenu = false,
		open = $bindable(true),
		revealChevronOnHover = false,
		trigger,
		triggerClass = '',
		triggerPosition = 'top',
		triggerStyle = ''
	}: Props = $props();

	// rows stay mounted through the collapse transition so it can play; after it
	// they unmount - mounted rows are menu items and would pollute the arrow-key
	// navigation
	// the region animates between height 0 and auto; see the component styles
	const EXPAND_TRANSITION_MS = 200;
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

{#snippet chevron()}
	<span
		class="ml-auto shrink-0 text-muted-foreground {open && revealChevronOnHover
			? 'opacity-0 group-hover:opacity-100'
			: ''}"
	>
		{#if open}
			<ChevronUp class={ICON_CLASS_DEFAULT} />
		{:else}
			<ChevronDown class={ICON_CLASS_DEFAULT} />
		{/if}
	</span>
{/snippet}

{#snippet triggerButton()}
	{#if inMenu}
		<!-- A menu item (for keyboard nav) wrapping the trigger via `child`;
		     closeOnSelect keeps the menu open while the section toggles. -->
		<DropdownMenu.Item
			class="group w-full min-w-0 cursor-pointer items-center gap-2 rounded-md text-left text-sm"
			closeOnSelect={false}
		>
			{#snippet child({ props })}
				<!-- No `class` here: a static attribute would override the spread props.class. -->
				<button
					{...props}
					aria-expanded={open}
					onclick={() => (open = !open)}
					style={triggerStyle}
					type="button"
				>
					{@render trigger()}

					{@render chevron()}
				</button>
			{/snippet}
		</DropdownMenu.Item>
	{:else}
		<button
			aria-expanded={open}
			class="group {triggerClass}"
			onclick={() => (open = !open)}
			style={triggerStyle}
			type="button"
		>
			{@render trigger()}

			{@render chevron()}
		</button>
	{/if}
{/snippet}

{#snippet region()}
	<!-- Custom expand region instead of bits-ui Collapsible (whose conditional
	     rendering kills the transition). -->
	<div class="collapsible-region" data-expanded={open}>
		{#if contentMounted}
			<div class="collapsible-region-content">
				{@render children()}
			</div>
		{/if}
	</div>
{/snippet}

{#if triggerPosition === 'top'}
	{@render triggerButton()}

	{@render region()}
{:else}
	{@render region()}

	{@render triggerButton()}
{/if}

<style>
	/*
	 * The region animates between height 0 and height auto. `interpolate-size`
	 * lets the auto keyword take part in the interpolation, so the content needs
	 * no measured height to stay in sync with. Browsers without it fall back to
	 * an interpolating grid row, which snaps only in the oldest engines.
	 */
	.collapsible-region {
		height: 0;
		overflow: hidden;
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
			overflow: hidden;
		}
	}
</style>
