<!--
	A row of an InsetList with two targets: the row itself opens a page, and a control at its end
	(a switch) acts on the row without opening it.
-->
<script lang="ts">
	import { ChevronRight } from '@lucide/svelte';
	import type { Component, Snippet } from 'svelte';
	import { cn } from '../../ui/utils';
	import { ICON_CLASS_DEFAULT } from '../../../constants';

	interface Props {
		children: Snippet;
		class?: string;
		/** Draw the row as unavailable. */
		dimmed?: boolean;
		icon?: Component<{ class?: string }>;
		/** Drawn in place of `icon`, for an avatar or an image. */
		leading?: Snippet;
		onclick: () => void;
		/** The control at the row's end. */
		trailing: Snippet;
	}

	let {
		children,
		class: className,
		dimmed = false,
		icon: Icon,
		leading,
		onclick,
		trailing
	}: Props = $props();
</script>

<div class={cn('group/row flex min-h-12 items-center px-4', dimmed && 'opacity-50', className)}>
	<button
		class="flex min-h-12 min-w-0 flex-1 items-center gap-3 self-stretch text-left text-body"
		{onclick}
		type="button"
	>
		{#if leading}
			{@render leading()}
		{:else if Icon}
			<Icon class={cn(ICON_CLASS_DEFAULT, 'shrink-0')} />
		{/if}

		<span class="inset-list-row-body">
			{@render children()}

			<ChevronRight class="size-4 shrink-0 text-muted-foreground" />
		</span>
	</button>

	<!-- the divider runs on under the control -->
	<span
		class="flex items-center self-stretch border-t border-border/60 pl-3 group-first/row:border-t-0"
	>
		{@render trailing()}
	</span>
</div>
