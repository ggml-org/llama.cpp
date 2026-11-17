<script lang="ts">
	import MenuActionsList from './MenuActionsList.svelte';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import * as Drawer from '$lib/components/ui/drawer';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Separator } from '$lib/components/ui/separator';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { MenuKind } from '$lib/enums';
	import { deviceStore } from '$lib/stores';
	import type { MenuAction } from '$lib/types';
	import type { Component } from 'svelte';

	interface Props {
		triggerIcon: Component;
		triggerTooltip?: string;
		triggerClass?: string;
		actions: MenuAction[];
		align?: 'start' | 'center' | 'end';
		/** Anchors the menu to this element instead of its trigger, and keeps it a menu on a phone too, for one opened by a press-and-hold. */
		anchor?: HTMLElement | null;
		open?: boolean;
	}

	let {
		actions,
		align = 'end',
		anchor = null,
		open = $bindable(false),
		triggerClass = '',
		triggerIcon,
		triggerTooltip
	}: Props = $props();

	// a phone shows the same actions in a drawer, where a thumb reaches them and every
	// row is a touch target of its own
	let isMobile = $derived(deviceStore.isMobile && !anchor);

	function runAction(action: MenuAction, event: Event) {
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

						<span class="flex flex-1 flex-col">
							{action.label}

							<!-- a phone has no hover, so the hint sits under the label -->
							{#if action.tooltip}
								<span class="text-xs text-muted-foreground">{action.tooltip}</span>
							{/if}
						</span>

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

		<DropdownMenu.Content {align} class="z-[999999] w-48" customAnchor={anchor}>
			<MenuActionsList {actions} kind={MenuKind.DROPDOWN} onClose={() => (open = false)} />
		</DropdownMenu.Content>
	</DropdownMenu.Root>
{/if}

{#snippet iconComponent(IconComponent: Component, className: string)}
	<IconComponent class={className} />
{/snippet}
