<script lang="ts">
	import { ChevronLeft, CircleAlert, Loader2 } from '@lucide/svelte';
	import { CollapsibleSection } from '$lib/components/app';
	import { modelsStore } from '$lib/stores';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		/** Extra classes for the chevron, to line it up with a row's own control. */
		chevronClass?: string;
		/** Number shown next to the label, omitted when undefined. */
		count?: number;
		error?: boolean;
		/** Overrides the backend logo, used for sections without a backend. */
		icon?: Snippet;
		label: string;
		loading?: boolean;
		/** Renders the back control, for a drilled-in provider. */
		onBack?: () => void;
		/** Persists the open state under this id, e.g. a section key. */
		persistKey?: string;
		/** Start expanded; the manager collapses its hidden block. */
		defaultOpen?: boolean;
		revealChevronOnHover?: boolean;
		sectionHeaderClass?: string;
		/** Sticky class of the header, so a surface can shade it differently. */
		stickyClass?: string;
		sticky?: boolean;
	}

	let {
		chevronClass = '',
		children,
		count,
		defaultOpen = true,
		error = false,
		icon,
		label,
		loading = false,
		onBack,
		persistKey,
		revealChevronOnHover = false,
		sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground select-none',
		sticky = false,
		stickyClass = 'sticky z-10 bg-popover'
	}: Props = $props();

	let triggerClass = $derived(
		`${sectionHeaderClass} flex w-full cursor-pointer items-center gap-1.5 text-left${sticky ? ` ${stickyClass}` : ''}`
	);
	// surfaces without a search block fall back to a 0 sticky height
	let triggerStyle = $derived(sticky ? 'top: var(--dropdown-sticky-height, 0px)' : '');
</script>

<CollapsibleSection
	{chevronClass}
	defaultOpen={persistKey ? modelsStore.isGroupOpen(persistKey, defaultOpen) : defaultOpen}
	ontoggle={persistKey ? (open: boolean) => modelsStore.setGroupOpen(persistKey, open) : null}
	{revealChevronOnHover}
	{triggerClass}
	{triggerStyle}
>
	{#snippet trigger()}
		{#if onBack}
			<button
				aria-label="Back to all providers"
				class="-ml-1 inline-flex shrink-0 cursor-pointer items-center rounded-sm p-0.5 text-muted-foreground transition hover:bg-muted/60 hover:text-foreground"
				onclick={(event) => {
					// the back control must not collapse the list it is leaving
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
