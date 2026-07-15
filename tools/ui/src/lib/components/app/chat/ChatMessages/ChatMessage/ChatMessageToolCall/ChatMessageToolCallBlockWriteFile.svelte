<script lang="ts">
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SPIN } from '$lib/constants/css-classes';
	import { Loader2, Wrench, XCircle } from '@lucide/svelte';
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
	const isCodeStreaming = $derived(isStreaming && (isPending || isStreamingCall));
	const toolUi = $derived(getBuiltinToolUi(section.toolName));
	const toolIcon = $derived(showSpinner ? Loader2 : (toolUi?.icon ?? Wrench));
	const toolIconClass = $derived(showSpinner ? ICON_CLASS_SPIN : ICON_CLASS_DEFAULT);
	const mcpServerFavicon = $derived(mcpStore.getServerFaviconForTool(section.toolName));
	const iconUrl = $derived(
		!showSpinner && !toolUi?.icon && mcpServerFavicon ? mcpServerFavicon : null
	);

	type WriteFileMeta = {
		fileName: string;
		filePath: string;
		language: string;
		content: string;
		bytesWritten?: number;
		resultMessage?: string;
		errorMessage?: string;
	};

	function parseWriteFileMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined,
		toolResultString: string | undefined
	): WriteFileMeta | null {
		if (toolName !== BuiltInTool.WRITE_FILE || !toolArgsString) return null;

		const args = parsePartialJsonArgs(toolArgsString);
		if (!args) return null;

		const rawPath = args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;
		const content = typeof args.content === 'string' ? args.content : '';
		const language =
			getFileTypeByExtension(rawPath)?.replace(TEXT_LANGUAGE_PREFIX_REGEX, '') ?? DEFAULT_LANGUAGE;

		let bytesWritten: number | undefined;
		let resultMessage: string | undefined;
		let errorMessage: string | undefined;
		if (toolResultString) {
			try {
				const parsed: unknown = JSON.parse(toolResultString);
				if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
					const obj = parsed as Record<string, unknown>;
					if (Number.isFinite(Number(obj.bytes))) {
						bytesWritten = Number(obj.bytes);
					}
					if (typeof obj.result === 'string') {
						resultMessage = obj.result;
					}
					if (typeof obj.error === 'string') {
						errorMessage = obj.error;
					}
				}
			} catch {
				// Non-JSON success string; leave result fields unset.
			}
		}

		return {
			fileName,
			filePath: rawPath,
			language,
			content,
			bytesWritten,
			resultMessage,
			errorMessage
		};
	}

	const writeFileMeta = $derived(
		parseWriteFileMeta(section.toolName, section.toolArgs, section.toolResult)
	);

	const subtitle = $derived(
		showSpinner
			? 'executing...'
			: writeFileMeta?.errorMessage
				? 'failed'
				: isStreamingCall && !isStreaming
					? 'incomplete'
					: undefined
	);
</script>

{#snippet writeFileTitle()}
	<span class="text-muted-foreground">Write file </span>
	<span class="font-mono">{writeFileMeta?.filePath}</span>
	{#if writeFileMeta?.errorMessage}
		<span class="ml-1 text-xs italic text-muted-foreground/70">(failed)</span>
	{/if}
{/snippet}

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	title=""
	titleSnippet={writeFileTitle}
	{subtitle}
	{onToggle}
>
	{#if writeFileMeta?.errorMessage}
		<div
			class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
		>
			<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
			<span>{writeFileMeta.errorMessage}</span>
		</div>
	{:else if writeFileMeta}
		<SyntaxHighlightedCode
			code={writeFileMeta.content}
			language={writeFileMeta.language}
			maxHeight={MAX_HEIGHT_CODE_BLOCK}
			streaming={isCodeStreaming}
		/>
		<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
			{#if writeFileMeta.resultMessage}
				{writeFileMeta.resultMessage}{writeFileMeta.bytesWritten != null
					? '\u00A0\u00B7\u00A0'
					: ''}{/if}
			{#if writeFileMeta.bytesWritten != null}
				<span class="font-mono">{writeFileMeta.bytesWritten}</span>
				bytes
			{/if}
		</div>
	{/if}
</CollapsibleContentBlock>
