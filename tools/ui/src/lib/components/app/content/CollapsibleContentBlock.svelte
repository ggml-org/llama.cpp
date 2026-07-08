<script lang="ts">
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { cn } from '$lib/components/ui/utils';
	import { useThrottle } from '$lib/hooks/use-throttle.svelte';
	import { formatReasoningPreview } from '$lib/utils';
	import { config } from '$lib/stores/settings.svelte';
	import type { Snippet } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		open?: boolean;
		class?: string;
		icon?: Component;
		iconClass?: string;
		title?: string;
		titleSnippet?: Snippet;
		subtitle?: string;
		preview?: string;
		rawContent?: string;
		isStreaming?: boolean;
		shimmerTitle?: boolean;
		onToggle?: () => void;
		children: Snippet;
	}

	let {
		open = $bindable(false),
		class: className = '',
		icon: IconComponent,
		iconClass = 'h-4 w-4',
		title = '',
		titleSnippet,
		subtitle,
		preview,
		rawContent,
		isStreaming = false,
		shimmerTitle = false,
		onToggle,
		children
	}: Props = $props();

	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);

	let previewKey = useThrottle(() => rawContent ?? preview ?? '', 500);
	let displayedPreview = $state('');
	let displayedOverflow = $state(0);

	$effect(() => {
		void previewKey.key;
		const content = rawContent ?? preview ?? '';
		const result = formatReasoningPreview(content);
		displayedPreview = result.preview;
		displayedOverflow = result.overflow;
	});
</script>

<Collapsible.Root
	{open}
	onOpenChange={(value) => {
		open = value;
		onToggle?.();
	}}
	class="my-0! {className} group/collapsible"
>
	<Collapsible.Trigger
		class="flex w-full cursor-pointer items-center justify-between gap-2 py-1.5 pr-1 text-left"
	>
		<div class="flex min-w-0 items-center gap-2 text-muted-foreground">
			{#if IconComponent}
				<IconComponent class={cn('shrink-0 text-muted-foreground/60', iconClass)} />
			{/if}

			<span class={cn('text-sm font-medium', shimmerTitle ? 'shimmer-text' : 'text-foreground/80')}>
				{#if titleSnippet}
					{@render titleSnippet()}
				{:else}
					{title}
				{/if}
			</span>

			{#if subtitle}
				<span class="text-xs italic text-muted-foreground/70">{subtitle}</span>
			{/if}

			{#if displayedPreview && !showThoughtInProgress && isStreaming}
				<div class="ml-1 flex min-w-0 items-baseline gap-2">
					<div class="w-3/4 truncate text-xs text-muted-foreground/60">
						{displayedPreview}
					</div>

					{#if displayedOverflow > 0}
						<span class="shrink-0 text-xs text-muted-foreground/40">+{displayedOverflow}</span>
					{/if}
				</div>
			{/if}
		</div>

		<ChevronDown
			class={cn(
				'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100',
				open && 'rotate-180'
			)}
		/>

		<span class="sr-only">Toggle content</span>
	</Collapsible.Trigger>

	<Collapsible.Content>
		<div class="pl-1.5 grid min-w-0" style="min-height: var(--min-message-height);">
			<div class="min-w-0 border-l border-muted-foreground/20 pl-4 pb-2 my-2">
				{@render children()}
			</div>
		</div>
	</Collapsible.Content>
</Collapsible.Root>
