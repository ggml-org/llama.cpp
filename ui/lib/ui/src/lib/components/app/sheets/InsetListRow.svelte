<!--
	A row of an InsetList, a finger tall and tappable as a whole: a button, a label around a switch,
	or a plain row holding a control. Its content is the row's body, after an optional icon.
-->
<script lang="ts">
	import { ChevronRight } from '@lucide/svelte';
	import type { ClassValue } from 'clsx';
	import type { Component, Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '../../ui/utils';
	import { ICON_CLASS_DEFAULT } from '../../../constants';

	interface Props extends Omit<HTMLAttributes<HTMLElement>, 'class'> {
		/** `button` is tapped, `label` wraps a switch, `div` holds a control of its own. */
		as?: 'button' | 'div' | 'label';
		/** Classes of the body: the label and value, whose top border is the divider. */
		bodyClass?: ClassValue;
		children: Snippet;
		/** Show a chevron at the row's end: the row opens a page. */
		chevron?: boolean;
		class?: ClassValue;
		disabled?: boolean;
		/** The icon before the body, outside the divider. */
		icon?: Component<{ class?: string }>;
		iconClass?: ClassValue;
		/** Drawn in place of `icon`, for an avatar or an image. */
		leading?: Snippet;
	}

	let {
		as = 'button',
		bodyClass,
		children,
		chevron = false,
		class: className,
		icon: Icon,
		iconClass,
		leading,
		...rest
	}: Props = $props();
</script>

<svelte:element
	this={as}
	class={cn('inset-list-row', as === 'label' && 'cursor-pointer', className)}
	type={as === 'button' ? 'button' : undefined}
	{...rest}
>
	{#if leading}
		{@render leading()}
	{:else if Icon}
		<Icon class={cn(ICON_CLASS_DEFAULT, 'shrink-0', iconClass)} />
	{/if}

	<span class={cn('inset-list-row-body', bodyClass)}>
		{@render children()}

		{#if chevron}
			<ChevronRight class="size-4 shrink-0 text-muted-foreground" />
		{/if}
	</span>
</svelte:element>
