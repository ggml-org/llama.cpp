<script lang="ts">
	import { InsetListRow } from '@ggml-org/llama-ui/app/sheets';
	import { Button, type ButtonVariant } from '@ggml-org/llama-ui/ui/button';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';
	import type { Component } from 'svelte';

	let {
		buttonClass,
		buttonText,
		buttonVariant,
		description,
		IconComponent,
		inset = false,
		onclick,
		summary,
		title,
		titleClass,
		wrapperClass
	}: {
		title: string;
		description: string;
		IconComponent: Component;
		/** Draw it as a row of a phone's inset list, its action as the row's name. */
		inset?: boolean;
		buttonText: string;
		onclick: () => void;
		titleClass?: string;
		buttonVariant?: ButtonVariant;
		buttonClass?: string;
		wrapperClass?: string;
		summary?: { show: boolean; verb: string; items: DatabaseConversation[] };
	} = $props();

	let sectionButtonClass = $derived(buttonClass ?? 'justify-start justify-self-start md:w-auto');
	let sectionButtonVariant = $derived(buttonVariant ?? 'outline');
	let isDestructive = $derived(buttonVariant === 'destructive');
</script>

{#snippet summaryList(summary: { verb: string; items: DatabaseConversation[] })}
	<h5 class="mb-2 text-label">
		{summary.verb}
		{summary.items.length} conversation{summary.items.length === 1 ? '' : 's'}
	</h5>

	<ul class="space-y-1 text-body-sm text-muted-foreground">
		{#each summary.items.slice(0, 10) as conv (conv.id)}
			<li class="truncate">• {conv.name || 'Untitled conversation'}</li>
		{/each}

		{#if summary.items.length > 10}
			<li class="italic">... and {summary.items.length - 10} more</li>
		{/if}
	</ul>
{/snippet}

{#if inset}
	<InsetListRow
		bodyClass="flex-col items-stretch gap-0.5 py-3"
		icon={IconComponent}
		iconClass={['mt-3.5 self-start', isDestructive && 'text-destructive']}
		{onclick}
	>
		<span class={isDestructive ? 'text-destructive' : undefined}>{buttonText}</span>

		<span class="text-caption text-muted-foreground">{description}</span>

		{#if summary && summary.show && summary.items.length > 0}
			<span class="mt-2 block">{@render summaryList(summary)}</span>
		{/if}
	</InsetListRow>
{:else}
	<div class="grid gap-1 {wrapperClass ?? ''}">
		<h4 class="mt-0 mb-2 text-label {titleClass ?? ''}">{title}</h4>

		<p class="mb-4 text-body-sm text-muted-foreground">{description}</p>

		<Button class={sectionButtonClass} {onclick} variant={sectionButtonVariant}>
			<IconComponent class="mr-2 {ICON_CLASS_DEFAULT}" />

			{buttonText}
		</Button>

		{#if summary && summary.show && summary.items.length > 0}
			<div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
				{@render summaryList(summary)}
			</div>
		{/if}
	</div>
{/if}
