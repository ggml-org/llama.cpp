<script lang="ts">
	/**
	 * CollapsibleInfoCard - Reusable collapsible card component
	 *
	 * Used for displaying thinking content, tool calls, and other collapsible information
	 * with a consistent UI pattern.
	 */
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';
	import type { Snippet } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		/** Whether the card is expanded */
		open?: boolean;
		/** CSS class for the root element */
		class?: string;
		/** Icon component to display */
		icon?: Component;
		/** Custom icon class (for animations like spin) */
		iconClass?: string;
		/** Title text */
		title: string;
		/** Optional subtitle/status text */
		subtitle?: string;
		/** Optional click handler for the trigger */
		onToggle?: () => void;
		/** Content to display in the collapsible section */
		children: Snippet;
	}

	let {
		open = $bindable(false),
		class: className = '',
		icon: Icon,
		iconClass = 'h-4 w-4',
		title,
		subtitle,
		onToggle,
		children
	}: Props = $props();
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
				class="overflow-y-auto border-t border-muted px-3 pb-3"
				style="max-height: calc(100dvh - var(--chat-form-area-height));"
			>
				{@render children()}
			</div>
		</Collapsible.Content>
	</Card>
</Collapsible.Root>
