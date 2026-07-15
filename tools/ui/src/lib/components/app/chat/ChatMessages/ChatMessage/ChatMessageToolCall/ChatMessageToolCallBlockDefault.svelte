<script lang="ts">
	import { Loader2, Wrench } from '@lucide/svelte';
	import {
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { AgenticSectionType, FileTypeText, ToolResultKind } from '$lib/enums';
	import { MAX_HEIGHT_CODE_BLOCK } from '$lib/constants';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import {
		classifyToolResult,
		formatJsonPretty,
		parseToolResultWithImages,
		type AgenticSection,
		type ToolResultLine
	} from '$lib/utils';
	import type { DatabaseMessageExtra } from '$lib/types';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let { section, open, isStreaming, attachments, onToggle }: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming));
	// True while the LLM is still emitting chunks into this tool call's args.
	// PENDING and STREAMING both cover this (chat-streaming tool calls are
	// surfaced as PENDING because toolArgs is partial while the outer call list
	// parses intact).
	const isCodeStreaming = $derived(isStreaming && (isPending || isStreamingCall));

	const toolUi = $derived(getBuiltinToolUi(section.toolName));
	const toolIcon = $derived(showSpinner ? Loader2 : (toolUi?.icon ?? Wrench));
	const toolIconClass = $derived(showSpinner ? 'h-4 w-4 animate-spin' : 'h-4 w-4');

	// Server favicon fallback for MCP tools with no built-in icon.
	const mcpServerFavicon = $derived(mcpStore.getServerFaviconForTool(section.toolName));
	const iconUrl = $derived(
		!showSpinner && !toolUi?.icon && mcpServerFavicon ? mcpServerFavicon : null
	);

	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithImages(section.toolResult, attachments) : []
	);

	// Pick a richer renderer (JSON / markdown) than the line-by-line fallback.
	const outputKind = $derived(classifyToolResult(section.toolResult));

	const title = $derived(toolUi?.label ?? section.toolName ?? '');

	const subtitle = $derived(
		showSpinner ? 'executing...' : isStreamingCall && !isStreaming ? 'incomplete' : undefined
	);
</script>

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	{title}
	{subtitle}
	{onToggle}
>
	{#if isStreamingCall}
		<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
			<span>Input</span>
			{#if isStreaming}
				<Loader2 class="h-3 w-3 animate-spin" />
			{/if}
		</div>
		{#if section.toolArgs}
			<SyntaxHighlightedCode
				code={formatJsonPretty(section.toolArgs)}
				language={FileTypeText.JSON}
				maxHeight={MAX_HEIGHT_CODE_BLOCK}
				streaming={isCodeStreaming}
			/>
		{:else if isStreaming}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Receiving arguments...
			</div>
		{:else}
			<div class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400">
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
				streaming={isCodeStreaming}
			/>
		{/if}
		<div
			class={showInput
				? 'mt-4 mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'
				: 'mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'}
		>
			<span>Output</span>
			{#if isPending}
				<Loader2 class="h-3 w-3 animate-spin" />
			{/if}
		</div>
		{#if isPending}
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
						<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{line.text}</div>
						{#if line.image}
							<img
								src={line.image.base64Url}
								alt={line.image.name}
								class="mt-2 mb-2 h-auto max-w-full rounded-lg"
								loading="lazy"
							/>
						{/if}
					{/each}
				</div>
			{/if}
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
		{/if}
	{/if}
</CollapsibleContentBlock>
