<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { deviceStore } from '$lib/stores';
	import type { Component, Snippet } from 'svelte';
	import type { HTMLButtonAttributes } from 'svelte/elements';

	interface Props extends Omit<HTMLButtonAttributes, 'children'> {
		children?: Snippet;
		icon?: Component;
		iconClass?: string;
		/** What it does, read out in place of a visible label. */
		label: string;
		/** md (40px) for a phone's corner keys, sm (36px) for a key floating over the page. */
		size?: 'md' | 'sm';
		/** Shows the label as a tooltip where there is a pointer to hover with. */
		tooltip?: boolean;
	}

	let {
		children,
		class: className = '',
		icon: Icon,
		iconClass = 'size-6',
		label,
		size = 'md',
		tooltip = false,
		...rest
	}: Props = $props();
</script>

{#snippet key(props: Record<string, unknown> = {})}
	<button
		{...props}
		{...rest}
		aria-label={label}
		class={[
			'inline-flex shrink-0 cursor-pointer items-center justify-center rounded-full border border-border/20 bg-muted/60 text-accent-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-muted',
			size === 'md' ? 'size-10' : 'size-9',
			className
		]}
		type="button"
	>
		{#if Icon}
			<Icon class={iconClass} />
		{/if}

		{@render children?.()}
	</button>
{/snippet}

<!-- a round key over the page: the menu and context gauge keys at a phone's top corners, and the
     key back to the bottom of a chat -->
{#if tooltip && !deviceStore.isMobile}
	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				{@render key(props)}
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{label}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{:else}
	{@render key()}
{/if}
