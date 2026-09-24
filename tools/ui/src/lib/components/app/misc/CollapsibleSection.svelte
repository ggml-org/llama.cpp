<script lang="ts">
	import CollapsibleRegion from './CollapsibleRegion.svelte';
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
		/** Start expanded. The section owns the state, so a parent rebuild cannot
		 *  snap it back open. Bind `open` instead to control it from outside. */
		defaultOpen?: boolean;
		/** Controlled open state, for a caller that binds it. */
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
		defaultOpen = true,
		inMenu = false,
		open = $bindable(defaultOpen),
		revealChevronOnHover = false,
		trigger,
		triggerClass = '',
		triggerPosition = 'top',
		triggerStyle = ''
	}: Props = $props();
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
	{#snippet body()}
		{@render children()}
	{/snippet}

	<CollapsibleRegion {open}>{@render body()}</CollapsibleRegion>
{/snippet}

{#if triggerPosition === 'top'}
	{@render triggerButton()}

	{@render region()}
{:else}
	{@render region()}

	{@render triggerButton()}
{/if}
