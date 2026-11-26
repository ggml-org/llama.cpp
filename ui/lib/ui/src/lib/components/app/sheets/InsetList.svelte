<!--
	An inset list: a rounded group of rows, as a phone's settings list them, with an optional title
	above and caption below. Its rows are InsetListRow and InsetListSplitRow.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '../../ui/utils';

	interface Props extends Omit<HTMLAttributes<HTMLDivElement>, 'class' | 'title'> {
		/** A note below the group. */
		caption?: Snippet | string;
		children: Snippet;
		/** Classes of the group itself. */
		class?: string;
		/** The group's title above it. */
		title?: Snippet | string;
	}

	let { caption, children, class: className, title, ...rest }: Props = $props();
</script>

<div class="flex flex-col gap-1.5" {...rest}>
	{#if title}
		<h3 class="inset-list-title">
			{#if typeof title === 'string'}{title}{:else}{@render title()}{/if}
		</h3>
	{/if}

	<section class={cn('inset-list', className)}>
		{@render children()}
	</section>

	{#if caption}
		<p class="inset-list-caption">
			{#if typeof caption === 'string'}{caption}{:else}{@render caption()}{/if}
		</p>
	{/if}
</div>
