<script lang="ts">
	import { parseFileGlobSearchMeta } from './parsers/file-glob-search';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { XCircle } from '@lucide/svelte';
	import { toolsStore } from '$lib/stores';
	import type { AgenticSection } from '$lib/types';
	import { abbreviateHome } from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	const fileGlobMeta = $derived(parseFileGlobSearchMeta(section));
	const home = $derived(toolsStore.serverHome);
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={fileGlobMeta} {onToggle}>
	{#snippet titleSnippet()}
		{#if fileGlobMeta}
			<span class="text-muted-foreground"
				>{fileGlobMeta.include === '**' ? 'List files' : 'Search files'}&nbsp;</span
			>
			{#if fileGlobMeta.include !== '**'}
				<span class="font-mono">{fileGlobMeta.include}</span>
			{/if}
			<span class="text-muted-foreground">&nbsp;in&nbsp;</span>
			<span class="font-mono" title={fileGlobMeta.path}
				>{abbreviateHome(fileGlobMeta.path, home)}</span
			>
		{/if}
	{/snippet}

	{#snippet children(meta, ctx)}
		{#if ctx.isPending}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Searching...
			</div>
		{:else if meta?.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
			>
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
				<span>{meta.errorMessage}</span>
			</div>
		{:else if meta && meta.matches.length > 0}
			<div class="max-h-96 overflow-auto">
				{#each meta.matches as match, i (i)}
					<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{match}</div>
				{/each}
			</div>
			<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
				Total matches: <span class="font-mono">{meta.totalMatches ?? meta.matches.length}</span>
			</div>
		{:else}
			<div class="text-xs text-muted-foreground/70 italic">No matches</div>
			<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
				Total matches: <span class="font-mono">{meta?.totalMatches ?? 0}</span>
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
