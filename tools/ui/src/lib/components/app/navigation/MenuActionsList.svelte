<script lang="ts">
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import * as ContextMenu from '$lib/components/ui/context-menu';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { MenuKind } from '$lib/enums';
	import type { MenuAction } from '$lib/types';
	import type { Component } from 'svelte';

	interface Props {
		actions: MenuAction[];
		kind: MenuKind;
	}

	let { actions, kind }: Props = $props();

	// both kits wrap the same bits-ui menu parts, so the rows render alike
	let Menu = $derived(kind === MenuKind.CONTEXT ? ContextMenu : DropdownMenu);
</script>

{#each actions as action, index (action.label)}
	{#if action.separator && index > 0}
		<Menu.Separator />
	{/if}

	{#if action.tooltip}
		<!-- a disabled item takes no pointer events, so the wrapper holds the hover -->
		<Tooltip.Root>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<div {...props}>
						{@render menuItem(action)}
					</div>
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content class="z-[999999] max-w-64" side="left">
				<p>{action.tooltip}</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{:else}
		{@render menuItem(action)}
	{/if}
{/each}

{#snippet menuItem(action: MenuAction)}
	<Menu.Item
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
	</Menu.Item>
{/snippet}

{#snippet iconComponent(IconComponent: Component, className: string)}
	<IconComponent class={className} />
{/snippet}
