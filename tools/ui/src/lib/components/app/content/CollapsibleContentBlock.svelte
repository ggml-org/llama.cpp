<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { cn } from '$lib/components/ui/utils';
	import type { Snippet } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		open?: boolean;
		class?: string;
		icon?: Component;
		iconClass?: string;
		iconUrl?: string | null;
		title?: string;
		titleSnippet?: Snippet;
		subtitle?: string;
		shimmerTitle?: boolean;
		onToggle?: () => void;
		children: Snippet;
	}

	let {
		open = $bindable(false),
		class: className = '',
		icon: IconComponent,
		iconClass = ICON_CLASS_DEFAULT,
		iconUrl = null,
		title = '',
		titleSnippet,
		subtitle,
		shimmerTitle = false,
		onToggle,
		children
	}: Props = $props();

	function hideBrokenIcon(event: Event) {
		(event.currentTarget as HTMLImageElement).style.display = 'none';
	}
</script>

<Collapsible.Root
	{open}
	onOpenChange={(value) => {
		open = value;
		onToggle?.();
	}}
	class={cn('group/collapsible', 'my-0!', className)}
>
	<Collapsible.Trigger
		class={cn(
			'flex w-full cursor-pointer items-start justify-between gap-2 text-left',
			'py-1.5 pr-1'
		)}
	>
		<div class="flex min-w-0 items-start gap-2 text-muted-foreground">
			{#if iconUrl}
				<img
					src={iconUrl}
					alt=""
					class={cn('shrink-0 rounded-sm  mt-0.75', iconClass)}
					onerror={hideBrokenIcon}
				/>
			{:else if IconComponent}
				<IconComponent class={cn('shrink-0 text-muted-foreground/60 mt-0.75', iconClass)} />
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
		</div>

		<ChevronDown
			class={cn(
				'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100 mt-0.75',
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
