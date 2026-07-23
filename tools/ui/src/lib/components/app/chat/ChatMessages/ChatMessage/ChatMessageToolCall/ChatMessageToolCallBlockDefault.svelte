<script lang="ts">
	// Fall-through renderer for tool calls without a dedicated block.
	// Renders section.toolArgs / section.toolResult directly using the
	// shared chrome shell.

	import { Loader2 } from '@lucide/svelte';
	import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
	import { FileTypeText, ToolResultKind } from '$lib/enums';
	import { MAX_HEIGHT_CODE_BLOCK } from '$lib/constants';
	import {
		classifyToolResult,
		formatJsonPretty,
		parseToolResultWithMedia,
		type AgenticSection,
		type ToolResultLine
	} from '$lib/utils';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import type { DatabaseMessageExtra } from '$lib/types';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let { section, open, isStreaming, attachments, onToggle }: Props = $props();

	const title = $derived(getBuiltinToolUi(section.toolName)?.label ?? section.toolName ?? '');

	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithMedia(section.toolResult, attachments) : []
	);
	const outputKind = $derived(classifyToolResult(section.toolResult));
</script>

<ToolCallBlock {section} {open} {isStreaming} meta={null} {title} {onToggle}>
	{#snippet children(_meta, ctx)}
		{#if ctx.isStreamingCall}
			<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
				<span>Input</span>
				{#if ctx.isStreaming}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>
			{#if section.toolArgs}
				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs)}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{:else if ctx.isStreaming}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Receiving arguments...
				</div>
			{:else}
				<div
					class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
				>
					Response was truncated
				</div>
			{/if}
		{:else}
			{@const showInput = Boolean(section.toolArgs)}
			{#if showInput}
				<div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
					<span>Input</span>
				</div>
				<SyntaxHighlightedCode
					code={formatJsonPretty(section.toolArgs ?? '')}
					language={FileTypeText.JSON}
					maxHeight={MAX_HEIGHT_CODE_BLOCK}
					streaming={ctx.isCodeStreaming}
				/>
			{/if}
			<div
				class={showInput
					? 'mt-4 mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'
					: 'mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'}
			>
				<span>Output</span>
				{#if ctx.isPending}
					<Loader2 class="h-3 w-3 animate-spin" />
				{/if}
			</div>
			{#if ctx.isPending}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
					Waiting for result...
				</div>
			{:else if section.toolResult}
				{#if outputKind === ToolResultKind.JSON}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolResult)}
						language={FileTypeText.JSON}
						maxHeight={MAX_HEIGHT_CODE_BLOCK}
					/>
				{:else if outputKind === ToolResultKind.MARKDOWN}
					<MarkdownContent content={section.toolResult} {attachments} />
				{:else}
					<div class="overflow-auto">
						{#each parsedLines as line, i (i)}
							<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
								{line.text}
							</div>
							{#if line.media}
								{#if line.media.type === 'AUDIO'}
									<div class="mt-2 mb-2">
										<audio controls class="w-full rounded-lg">
											<source src={line.media.base64Url} type={line.media.mimeType ?? 'audio/mpeg'} />
											Your browser does not support the audio element.
										</audio>
									</div>
								{:else}
									<img
										src={line.media.base64Url}
										alt={line.media.name}
										class="mt-2 mb-2 h-auto max-w-full rounded-lg"
										loading="lazy"
									/>
									{/if}
								{/if}
						{/each}
					</div>
				{/if}
			{:else}
				<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
			{/if}
		{/if}
	{/snippet}
</ToolCallBlock>
