<script lang="ts">
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SPIN } from '$lib/constants/css-classes';
	import { Loader2, Wrench } from '@lucide/svelte';
	import { CollapsibleContentBlock, SyntaxHighlightedCode } from '$lib/components/app';
	import { AgenticSectionType, BuiltInTool } from '$lib/enums';
	import {
		DEFAULT_LANGUAGE,
		FILE_PATH_SEPARATOR_REGEX,
		MAX_HEIGHT_CODE_BLOCK,
		TEXT_LANGUAGE_PREFIX_REGEX
	} from '$lib/constants';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { getFileTypeByExtension, parsePartialJsonArgs, type AgenticSection } from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, open, isStreaming, onToggle }: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming));
	const toolUi = $derived(getBuiltinToolUi(section.toolName));
	const toolIcon = $derived(showSpinner ? Loader2 : (toolUi?.icon ?? Wrench));
	const toolIconClass = $derived(showSpinner ? ICON_CLASS_SPIN : ICON_CLASS_DEFAULT);
	const mcpServerFavicon = $derived(mcpStore.getServerFaviconForTool(section.toolName));
	const iconUrl = $derived(
		!showSpinner && !toolUi?.icon && mcpServerFavicon ? mcpServerFavicon : null
	);

	type ReadFileMeta = {
		fileName: string;
		lineRange: { start: number; end: number } | null;
		language: string;
	};

	function parseReadFileMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined
	): ReadFileMeta | null {
		if (toolName !== BuiltInTool.READ_FILE || !toolArgsString) return null;

		const args = parsePartialJsonArgs(toolArgsString);
		if (!args) return null;

		const rawPath = args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;

		const startRaw = args.start_line ?? args.line_start ?? args.startLine ?? args.from_line;
		const endRaw = args.end_line ?? args.line_end ?? args.endLine ?? args.to_line;
		const countRaw = args.line_count ?? args.count ?? args.num_lines;

		let lineRange: { start: number; end: number } | null = null;
		const sNum = Number(startRaw);
		const eNum = Number(endRaw);
		if (startRaw != null && endRaw != null && Number.isFinite(sNum) && Number.isFinite(eNum)) {
			lineRange = { start: sNum, end: eNum };
		} else if (startRaw != null && countRaw != null) {
			const cNum = Number(countRaw);
			if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
				lineRange = { start: sNum, end: sNum + cNum - 1 };
			}
		}

		const fileType = getFileTypeByExtension(fileName);
		const language = fileType ? fileType.replace(TEXT_LANGUAGE_PREFIX_REGEX, '') : DEFAULT_LANGUAGE;

		return { fileName, lineRange, language };
	}

	const readFileMeta = $derived(parseReadFileMeta(section.toolName, section.toolArgs));
</script>

{#snippet readFileTitle()}
	<span class="text-muted-foreground">Read file </span>
	<span class="font-mono">{readFileMeta?.fileName}</span>
	{#if readFileMeta?.lineRange}
		<span class="text-muted-foreground"
			>&nbsp;(lines {readFileMeta.lineRange.start}-{readFileMeta.lineRange.end})</span
		>
	{/if}
{/snippet}

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	title=""
	titleSnippet={readFileTitle}
	{onToggle}
>
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
</CollapsibleContentBlock>
