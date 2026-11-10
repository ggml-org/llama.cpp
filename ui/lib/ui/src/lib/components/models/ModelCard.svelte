<script lang="ts">
	import * as Card from '$lib/components/ui/card';
	import type { Snippet } from 'svelte';

	interface Props {
		/** Avatar at the card's top left. */
		avatar: Snippet;
		/** The model's name and badges, beside the avatar; the card's one button. */
		title: Snippet;
		/** Controls at the top right, e.g. an actions menu. */
		actions?: Snippet;
		/** Body between the header and the footer: capabilities, a description. */
		children?: Snippet;
		class?: string;
		/** Faded, e.g. a hidden model. */
		dimmed?: boolean;
		/** Muted facts at the bottom left. */
		footer?: Snippet;
		/** Bottom right of the footer, e.g. a load control. */
		footerEnd?: Snippet;
		onSelect?: () => void;
		/** Laid over the card's edge, e.g. a download progress bar. */
		overlay?: Snippet;
		/** `aria-pressed` of the card's button; undefined leaves it a plain button. */
		pressed?: boolean;
		selected?: boolean;
		/** Native title of the card's button, e.g. the full model id. */
		tooltip?: string;
		/** Makes the card a link instead of a button, for pages where a model has a page of its own (llama-ui only). */
		href?: string;
	}

	let {
		actions,
		avatar,
		children,
		class: className = '',
		dimmed = false,
		footer,
		footerEnd,
		href,
		onSelect,
		overlay,
		pressed,
		selected = false,
		title,
		tooltip
	}: Props = $props();
</script>

<!-- The surface of the MCP server cards. The title button (or link) stretches over the whole card,
     so a click anywhere opens the model, while the controls stack above it: a button
     nested in a button is invalid. -->
<Card.Root
	class={[
		'relative min-w-0 gap-3! overflow-hidden bg-muted/30 p-4 transition-colors hover:bg-muted/50',
		selected && 'ring-1 ring-primary/40',
		dimmed && 'opacity-60',
		className
	]}
>
	<div class="flex min-w-0 items-start gap-3">
		<svelte:element
			this={href ? 'a' : 'button'}
			aria-pressed={href ? undefined : pressed}
			class="flex min-w-0 flex-1 cursor-pointer items-center gap-3 text-left before:absolute before:inset-0 before:content-[''] focus-visible:outline-none focus-visible:before:rounded-xl focus-visible:before:ring-2 focus-visible:before:ring-ring/50"
			{href}
			onclick={onSelect}
			role={href ? 'link' : 'button'}
			title={tooltip}
			type={href ? undefined : 'button'}
		>
			{@render avatar()}

			<span class="min-w-0 flex-1">
				{@render title()}
			</span>
		</svelte:element>

		{#if actions}
			<div class="relative z-10 -mt-1 -mr-1.5 shrink-0">
				{@render actions()}
			</div>
		{/if}
	</div>

	{@render children?.()}

	{#if footer || footerEnd}
		<div class="mt-auto flex min-h-6 items-center justify-between gap-3 text-xs">
			<div class="min-w-0 text-muted-foreground">
				{@render footer?.()}
			</div>

			{#if footerEnd}
				<div class="relative z-10 flex shrink-0 items-center">
					{@render footerEnd()}
				</div>
			{/if}
		</div>
	{/if}

	{@render overlay?.()}
</Card.Root>
