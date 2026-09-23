<script lang="ts">
	import { ChevronLeft, CircleAlert, Loader2 } from '@lucide/svelte';
	import { CollapsibleSection } from '$lib/components/app';
	import { BackendIcon } from '$lib/components/app/backends';
	import { getBackend } from '$lib/utils/api-base';
	import type { Snippet } from 'svelte';

	interface Props {
		/** Section rows. */
		children: Snippet;
		/** Renders the backend's logo in the header. */
		backendId?: string;
		/** Number shown next to the label, omitted when undefined. */
		count?: number;
		error?: boolean;
		/** Overrides the backend logo, used for sections without a backend. */
		icon?: Snippet;
		label: string;
		loading?: boolean;
		/** Renders the back control, for a drilled-in provider. */
		onBack?: () => void;
		/** Start expanded; the manager collapses its hidden block. */
		open?: boolean;
		revealChevronOnHover?: boolean;
		sectionHeaderClass?: string;
		/** Sticks the header to the top of the scrollport. */
		sticky?: boolean;
	}

	let {
		backendId,
		children,
		count,
		error = false,
		icon,
		label,
		loading = false,
		onBack,
		open = true,
		revealChevronOnHover = false,
		sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground select-none',
		sticky = false
	}: Props = $props();

	let triggerClass = $derived(
		`${sectionHeaderClass} flex w-full cursor-pointer items-center gap-1.5 text-left${sticky ? ' sticky z-10 bg-popover' : ''}`
	);
	// the dropdown publishes its search block height, surfaces without one fall back to 0
	let triggerStyle = $derived(sticky ? 'top: var(--dropdown-sticky-height, 0px)' : '');
</script>

<CollapsibleSection {open} {revealChevronOnHover} {triggerClass} {triggerStyle}>
	{#snippet trigger()}
		{#if onBack}
			<button
				aria-label="Back to all providers"
				class="-ml-1 inline-flex shrink-0 cursor-pointer items-center rounded-sm p-0.5 text-muted-foreground transition hover:bg-muted/60 hover:text-foreground"
				onclick={(event) => {
					// the surrounding trigger toggles the section, the back control
					// must not collapse the list it is leaving
					event.stopPropagation();
					onBack?.();
				}}
				type="button"
			>
				<ChevronLeft class="h-3.5 w-3.5" />
			</button>
		{/if}

		{#if icon}
			{@render icon()}
		{:else if backendId}
			<BackendIcon backend={getBackend(backendId)} class="h-3.5 w-3.5" />
		{/if}

		<span class="truncate">{label}</span>

		{#if loading}
			<Loader2 class="h-3 w-3 shrink-0 animate-spin" />
		{:else if error}
			<CircleAlert class="h-3 w-3 shrink-0 text-destructive" />
		{/if}

		{#if count !== undefined}
			<span class="shrink-0 text-xs text-muted-foreground/70">{count}</span>
		{/if}
	{/snippet}

	{@render children()}
</CollapsibleSection>
