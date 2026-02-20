<script lang="ts">
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';
	import { createAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
	import type { Snippet } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		open?: boolean;
		class?: string;
		icon?: Component;
		iconClass?: string;
		title: string;
		subtitle?: string;
		isStreaming?: boolean;
		onToggle?: () => void;
		children: Snippet;
	}

	let {
		open = $bindable(false),
		class: className = '',
		icon: Icon,
		iconClass = 'h-4 w-4',
		title,
		subtitle,
		isStreaming = false,
		onToggle,
		children
	}: Props = $props();

	let contentContainer: HTMLDivElement | undefined = $state();
	const autoScroll = createAutoScrollController();

	$effect(() => {
		autoScroll.setContainer(contentContainer);
	});

	$effect(() => {
		// Only auto-scroll when open and streaming
		autoScroll.updateInterval(open && isStreaming);
	});

	function handleScroll() {
		autoScroll.handleScroll();
	}
</script>

<Collapsible.Root
	{open}
	onOpenChange={(value) => {
		open = value;
		onToggle?.();
	}}
	class={className}
>
	<Card class="gap-0 border-muted bg-muted/30 py-0">
		<Collapsible.Trigger class="flex w-full cursor-pointer items-center justify-between p-3">
			<div class="flex items-center gap-2 text-muted-foreground">
				{#if Icon}
					<Icon class={iconClass} />
				{/if}

				<span class="font-mono text-sm font-medium">{title}</span>

				{#if subtitle}
					<span class="text-xs italic">{subtitle}</span>
				{/if}
			</div>

			<div
				class={buttonVariants({
					variant: 'ghost',
					size: 'sm',
					class: 'h-6 w-6 p-0 text-muted-foreground hover:text-foreground'
				})}
			>
				<ChevronsUpDownIcon class="h-4 w-4" />

				<span class="sr-only">Toggle content</span>
			</div>
		</Collapsible.Trigger>

		<Collapsible.Content>
			<div
				bind:this={contentContainer}
				class="overflow-y-auto border-t border-muted px-3 pb-3"
				onscroll={handleScroll}
				style="min-height: var(--min-message-height); max-height: var(--max-message-height);"
			>
				{@render children()}
			</div>
		</Collapsible.Content>
	</Card>
</Collapsible.Root>
