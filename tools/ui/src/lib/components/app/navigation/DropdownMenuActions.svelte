<script lang="ts">
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import * as Drawer from '$lib/components/ui/drawer';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Separator } from '$lib/components/ui/separator';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { deviceStore } from '$lib/stores';
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
		actions,
		align = 'end',
		open = $bindable(false),
		triggerClass = '',
		triggerIcon,
		triggerTooltip
	}: Props = $props();

	// a phone shows the same actions in a drawer, where a thumb reaches them and every
	// row is a touch target of its own
	let isMobile = $derived(deviceStore.isMobile);

	function runAction(action: ActionItem, event: Event) {
		open = false;

		// let the drawer close before an action that moves the focus, such as a confirm dialog
		setTimeout(() => action.onclick(event), 0);
	}
</script>

{#if isMobile}
	<button
		class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {triggerClass}"
		data-slot="dropdown-menu-trigger"
		data-state={open ? 'open' : 'closed'}
		onclick={(event) => {
			event.stopPropagation();
			open = true;
		}}
		type="button"
	>
		{@render iconComponent(triggerIcon, 'h-3 w-3')}

		{#if triggerTooltip}
			<span class="sr-only">{triggerTooltip}</span>
		{/if}
	</button>

	<Drawer.Root bind:open>
		<Drawer.Content>
			<Drawer.Header>
				<Drawer.Title>{triggerTooltip ?? 'Actions'}</Drawer.Title>
			</Drawer.Header>

			<div class="flex flex-col px-2 pb-4">
				{#each actions as action, index (action.label)}
					{#if action.separator && index > 0}
						<Separator class="my-1" />
					{/if}

					<button
						class="flex min-h-11 w-full cursor-pointer items-center gap-3 rounded-md px-3 text-left text-sm transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50 {action.variant ===
						'destructive'
							? 'text-destructive'
							: ''}"
						disabled={action.disabled}
						onclick={(event) => runAction(action, event)}
						type="button"
					>
						{@render iconComponent(action.icon, 'h-4 w-4 shrink-0')}

						<span class="flex-1">{action.label}</span>

						{#if action.shortcut}
							<KeyboardShortcutInfo keys={action.shortcut} variant={action.variant} />
						{/if}
					</button>
				{/each}
			</div>
		</Drawer.Content>
	</Drawer.Root>
{:else}
	<DropdownMenu.Root bind:open>
		<Tooltip.Root>
			<Tooltip.Trigger>
				<!-- prevent another nested button element -->
				{#snippet child({ props })}
					<DropdownMenu.Trigger
						{...props}
						class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {triggerClass}"
						onclick={(e) => e.stopPropagation()}
					>
						{@render iconComponent(triggerIcon, 'h-3 w-3')}

						{#if triggerTooltip}
							<span class="sr-only">{triggerTooltip}</span>
						{/if}
					</DropdownMenu.Trigger>
				{/snippet}
			</Tooltip.Trigger>

			{#if triggerTooltip}
				<Tooltip.Content>
					<p>{triggerTooltip}</p>
				</Tooltip.Content>
			{/if}
		</Tooltip.Root>

		<DropdownMenu.Content {align} class="z-[999999] w-48">
			{#each actions as action, index (action.label)}
				{#if action.separator && index > 0}
					<DropdownMenu.Separator />
				{/if}

				<DropdownMenu.Item
					class="flex items-center justify-between hover:[&>kbd]:opacity-100"
					disabled={action.disabled}
					onclick={action.onclick}
					variant={action.variant}
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
{/if}

{#snippet iconComponent(IconComponent: Component, className: string)}
	<IconComponent class={className} />
{/snippet}
