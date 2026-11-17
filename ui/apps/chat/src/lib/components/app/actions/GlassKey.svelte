<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { GLASS_KEY_CLASS } from '#lib/constants/index.ts';
	import { deviceStore } from '#lib/stores/index.ts';
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
		class={[GLASS_KEY_CLASS, size === 'md' ? 'size-10' : 'size-9', className]}
		type="button"
	>
		{#if Icon}
			<Icon class={iconClass} />
		{/if}

		{@render children?.()}
	</button>
{/snippet}

<!-- a round matte glass key: the menu and context gauge keys at a phone's top corners, and the
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
