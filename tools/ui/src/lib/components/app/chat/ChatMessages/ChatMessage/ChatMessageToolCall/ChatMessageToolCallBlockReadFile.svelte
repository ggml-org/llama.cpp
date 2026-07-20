<script lang="ts">
	import { SyntaxHighlightedCode } from '$lib/components/app';
	import { DEFAULT_LANGUAGE, MAX_HEIGHT_CODE_BLOCK } from '$lib/constants';
	import { type AgenticSection } from '$lib/utils';
	import { parseReadFileMeta } from './parsers/read-file';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const readFileMeta = $derived(parseReadFileMeta(section));
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={readFileMeta} {onToggle}>
	{#snippet titleSnippet()}
		<span class="text-muted-foreground">Read file </span>
		<span class="font-mono">{readFileMeta?.fileName}</span>
		{#if readFileMeta?.lineRange}
			<span class="text-muted-foreground"
				>&nbsp;(lines {readFileMeta.lineRange.start}-{readFileMeta.lineRange.end})</span
			>
		{/if}
	{/snippet}

	{#snippet children(_meta, _ctx)}
		{#if section.toolResult}
			<SyntaxHighlightedCode
				code={section.toolResult}
				language={readFileMeta?.language ?? DEFAULT_LANGUAGE}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
			/>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for file content...
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
