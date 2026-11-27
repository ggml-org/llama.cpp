<script lang="ts">
	import { parseReadFileMeta } from './parsers/read-file';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import ToolCallTitle from './ToolCallTitle.svelte';
	import { SyntaxHighlightedCode } from '#lib/components/app/index.ts';
	import { CODE_BLOCK, MAX_HEIGHT_CODE_BLOCK } from '#lib/constants/index.ts';
	import type { AgenticSection } from '#lib/types/index.ts';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	const readFileMeta = $derived(parseReadFileMeta(section));
</script>

<ToolCallBlock {isStreaming} meta={readFileMeta} {onToggle} {open} {section}>
	{#snippet titleSnippet(phrase)}
		<ToolCallTitle label={phrase.label} target={readFileMeta?.fileName}>
			{#if readFileMeta?.lineRange}
				<span class="shrink-0 text-muted-foreground">
					(lines {readFileMeta.lineRange.start}-{readFileMeta.lineRange.end})
				</span>
			{/if}
		</ToolCallTitle>
	{/snippet}

	{#snippet children(_meta, _ctx)}
		{#if section.toolResult}
			<SyntaxHighlightedCode
				code={section.toolResult}
				language={readFileMeta?.language ?? CODE_BLOCK.DEFAULT_LANGUAGE}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
			/>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-caption text-muted-foreground/70 italic">
				Waiting for file content...
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
