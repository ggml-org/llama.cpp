<script lang="ts">
	import { parseWriteFileMeta, parseWriteFileTitleMeta } from './parsers/write-file';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import ToolCallTitle from './ToolCallTitle.svelte';
	import { XCircle } from '@lucide/svelte';
	import { SyntaxHighlightedCode } from '#lib/components/app/index.ts';
	import { MAX_HEIGHT_CODE_BLOCK, RESULT_STAT_SEPARATOR } from '#lib/constants/index.ts';
	import { FileTypeText } from '#lib/enums/index.ts';
	import { toolsStore } from '#lib/stores/index.ts';
	import type { AgenticSection } from '#lib/types/index.ts';
	import { abbreviateHome, formatJsonPretty } from '#lib/utils/index.ts';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	const writeFileMeta = $derived(parseWriteFileTitleMeta(section));
	// body-only: the full meta parses the embedded file content, and this
	// derived is read solely from the children snippet, which renders only
	// while the block is expanded
	const writeFileBody = $derived(parseWriteFileMeta(section));
	const home = $derived(toolsStore.serverHome);
</script>

<ToolCallBlock {isStreaming} meta={writeFileMeta} {onToggle} {open} {section}>
	{#snippet titleSnippet(phrase)}
		<ToolCallTitle
			label={phrase.label}
			target={abbreviateHome(writeFileMeta?.filePath ?? '', home)}
			targetTitle={writeFileMeta?.filePath}
		>
			{#if writeFileMeta?.errorMessage}
				<span class="shrink-0 text-caption italic text-muted-foreground/70">(failed)</span>
			{/if}
		</ToolCallTitle>
	{/snippet}

	{#snippet children(_meta, ctx)}
		{#if writeFileBody?.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-surface-danger p-2 text-caption text-fg-danger italic"
			>
				<XCircle class="mt-0.5 size-icon-xs shrink-0" />

				<span>{writeFileBody.errorMessage}</span>
			</div>
		{:else if writeFileBody?.content !== undefined}
			<SyntaxHighlightedCode
				code={writeFileBody.content}
				language={writeFileBody.language}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
				streaming={ctx.isCodeStreaming}
			/>

			<div class="mt-1.5 text-caption text-muted-foreground/70 italic">
				{#if writeFileBody.resultMessage}
					{writeFileBody.resultMessage}{writeFileBody.bytesWritten != null
						? RESULT_STAT_SEPARATOR
						: ''}{/if}

				{#if writeFileBody.bytesWritten != null}
					<span class="font-mono">{writeFileBody.bytesWritten}</span>
					bytes
				{/if}
			</div>
		{:else if section.toolArgs}
			<!-- no content arg: not streamed yet, or a same-named tool with other arg names -->
			<SyntaxHighlightedCode
				code={formatJsonPretty(section.toolArgs)}
				language={FileTypeText.JSON}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
				streaming={ctx.isCodeStreaming}
			/>
		{:else if ctx.isStreaming}
			<div class="rounded bg-muted/20 p-2 text-caption text-muted-foreground/70 italic">
				Receiving arguments...
			</div>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-caption text-muted-foreground/70 italic">
				No content
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
