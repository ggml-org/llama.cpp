<script lang="ts">
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';
	import type { Component } from 'svelte';

	interface ActionItem {
		icon: Component;
		label: string;
		onclick: (event: Event) => void;
		variant?: 'default' | 'destructive';
		disabled?: boolean;
		shortcut?: string[];
		separator?: boolean;
	}

	interface Props {
		triggerIcon: Component;
		triggerTooltip?: string;
		triggerClass?: string;
		actions: ActionItem[];
		align?: 'start' | 'center' | 'end';
		open?: boolean;
	}

	let {
		triggerIcon,
		triggerTooltip,
		triggerClass = '',
		actions,
		align = 'end',
		open = $bindable(false)
	}: Props = $props();
</script>

<DropdownMenu.Root bind:open>
	<DropdownMenu.Trigger
		class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {triggerClass}"
		onclick={(e) => e.stopPropagation()}
	>
		{#if triggerTooltip}
			<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
				<Tooltip.Trigger>
					{@render iconComponent(triggerIcon, 'h-3 w-3')}
					<span class="sr-only">{triggerTooltip}</span>
				</Tooltip.Trigger>
				<Tooltip.Content>
					<p>{triggerTooltip}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{:else}
			{@render iconComponent(triggerIcon, 'h-3 w-3')}
		{/if}
	</DropdownMenu.Trigger>

	<DropdownMenu.Content {align} class="z-[999999] w-48">
		{#each actions as action, index (action.label)}
			{#if action.separator && index > 0}
				<DropdownMenu.Separator />
			{/if}

			<DropdownMenu.Item
				onclick={action.onclick}
				variant={action.variant}
				disabled={action.disabled}
				class="flex items-center justify-between hover:[&>kbd]:opacity-100"
			>
				<div class="flex items-center gap-2">
					{@render iconComponent(
						action.icon,
						`h-4 w-4 ${action.variant === 'destructive' ? 'text-destructive' : ''}`
					)}
					{action.label}
				</div>

				{#if action.shortcut}
					<KeyboardShortcutInfo keys={action.shortcut} variant={action.variant} />
				{/if}
			</DropdownMenu.Item>
		{/each}
	</DropdownMenu.Content>
</DropdownMenu.Root>

{#snippet iconComponent(IconComponent: Component, className: string)}
	<IconComponent class={className} />
{/snippet}
