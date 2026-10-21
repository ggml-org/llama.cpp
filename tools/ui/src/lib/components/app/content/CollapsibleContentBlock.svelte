<script lang="ts">
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { cn } from '$lib/components/ui/utils';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
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
		/**
		 * `divider` centers the toggle between two rules, for blocks that mark a
		 * section boundary in the conversation instead of decorating a message.
		 */
		variant?: 'default' | 'divider';
		/**
		 * Render the header only, without a chevron or a body, for a block that
		 * has nothing to show yet and turns into a collapsible once it has.
		 */
		collapsible?: boolean;
		onToggle?: () => void;
		children: Snippet;
	}

	let {
		children,
		class: className = '',
		collapsible = true,
		icon: IconComponent,
		iconClass = ICON_CLASS_DEFAULT,
		iconUrl = null,
		onToggle,
		open = $bindable(false),
		shimmerTitle = false,
		subtitle,
		title = '',
		titleSnippet,
		variant = 'default'
	}: Props = $props();

	function hideBrokenIcon(event: Event) {
		(event.currentTarget as HTMLImageElement).style.display = 'none';
	}
</script>

{#snippet toggleIcon(layoutClass: string)}
	{#if iconUrl}
		<img
			alt=""
			class={cn('shrink-0 rounded-sm', layoutClass, iconClass)}
			onerror={hideBrokenIcon}
			src={iconUrl}
		/>
	{:else if IconComponent}
		<IconComponent class={cn('shrink-0 text-muted-foreground/60', layoutClass, iconClass)} />
	{/if}
{/snippet}

{#snippet dividerRow()}
	<span aria-hidden="true" class="h-px min-w-4 flex-1 bg-current opacity-25"></span>

	<div class="flex min-w-0 items-center gap-1.5">
		{@render toggleIcon('')}

		{@render toggleTitle('text-xs whitespace-nowrap')}

		{#if collapsible}
			<ChevronDown
				class={cn(
					'size-3.5 shrink-0 text-muted-foreground/60 transition-transform duration-150 ease-out',
					open && 'rotate-180'
				)}
			/>
		{/if}
	</div>

	<span aria-hidden="true" class="h-px min-w-4 flex-1 bg-current opacity-25"></span>
{/snippet}

{#snippet toggleTitle(layoutClass: string)}
	<span
		class={cn(
			'min-w-0 font-medium',
			layoutClass,
			shimmerTitle ? 'shimmer-text' : 'text-foreground/80'
		)}
	>
		{#if titleSnippet}
			{@render titleSnippet()}
		{:else}
			{title}
		{/if}
	</span>

	{#if subtitle}
		<span class="text-xs italic text-muted-foreground/70">{subtitle}</span>
	{/if}
{/snippet}

<Collapsible.Root
	class={cn('group/collapsible', 'my-0!', className)}
	onOpenChange={(value) => {
		open = value;
		onToggle?.();
	}}
	{open}
>
	{#if variant === 'divider' && !collapsible}
		<div class="flex w-full items-center gap-2 py-1.5 text-muted-foreground">
			{@render dividerRow()}
		</div>
	{:else if variant === 'divider'}
		<Collapsible.Trigger
			class="flex w-full cursor-pointer items-center gap-2 py-1.5 text-muted-foreground"
		>
			{@render dividerRow()}

			<span class="sr-only">Toggle content</span>
		</Collapsible.Trigger>
	{:else}
		<Collapsible.Trigger
			class={cn(
				'flex w-full cursor-pointer items-start justify-between gap-2 text-left',
				'py-1.5 pr-1'
			)}
		>
			<div class="flex min-w-0 items-start gap-2 text-muted-foreground">
				{@render toggleIcon('mt-0.75')}

				{@render toggleTitle('overflow-x-auto text-sm')}
			</div>

			<ChevronDown
				class={cn(
					'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100 mt-0.75',
					open && 'rotate-180'
				)}
			/>

			<span class="sr-only">Toggle content</span>
		</Collapsible.Trigger>
	{/if}

	{#if collapsible}
		<Collapsible.Content>
			<!-- Collapsible.Content renders its children unconditionally and only sets
			     `hidden`, so a closed block would keep re-rendering its whole body on
			     every streamed token. Gate on `open` so collapsed content costs nothing. -->
			{#if open}
				<div class="pl-1.5 grid min-w-0" style="min-height: var(--min-message-height);">
					<div class="min-w-0 border-l border-muted-foreground/20 pl-4 pb-2 my-2">
						{@render children()}
					</div>
				</div>
			{/if}
		</Collapsible.Content>
	{/if}
</Collapsible.Root>
