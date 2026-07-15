<script lang="ts">
	import { XCircle } from '@lucide/svelte';
	import { type AgenticSection } from '$lib/utils';
	import { parseGrepSearchMeta } from './parsers/grep-search';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const grepMeta = $derived(parseGrepSearchMeta(section));
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={grepMeta} {onToggle}>
	{#snippet titleSnippet()}
		{#if grepMeta}
			<span class="text-muted-foreground">Search for&nbsp;</span>
			<span class="font-mono">{grepMeta.pattern}</span>
			<span class="text-muted-foreground">&nbsp;in&nbsp;</span>
			<span class="font-mono">{grepMeta.path}</span>
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
				{#each meta.matches as match, mi (mi)}
					<div class="font-mono text-[11px] leading-relaxed">
						<span class="text-muted-foreground/70">{match.file}</span>
						{#if meta.showLineNumbers && match.line != null}
							<span class="text-muted-foreground/70">:{match.line}</span>
						{/if}
						<span class="text-muted-foreground/70">:</span>
						<span>{match.content}</span>
					</div>
				{/each}
			</div>
			<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
				Total matches: <span class="font-mono">{meta.totalMatches ?? meta.matches.length}</span>
				{#if meta.showLineNumbers}
					&nbsp;<span class="italic">(with line numbers)</span>
				{/if}
			</div>
		{:else}
			<div class="text-xs text-muted-foreground/70 italic">No matches</div>
			<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
				Total matches: <span class="font-mono">{meta?.totalMatches ?? 0}</span>
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
