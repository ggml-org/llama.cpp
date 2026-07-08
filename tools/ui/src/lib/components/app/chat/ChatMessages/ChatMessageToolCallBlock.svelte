<script lang="ts">
	import { Wrench, Loader2, XCircle } from '@lucide/svelte';
	import { CollapsibleContentBlock, SyntaxHighlightedCode } from '$lib/components/app';
	import { AgenticSectionType, FileTypeText } from '$lib/enums';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import {
		formatJsonPretty,
		getFileTypeByExtension,
		parseToolResultWithImages,
		type AgenticSection,
		type ToolResultLine
	} from '$lib/utils';
	import type { DatabaseMessageExtra } from '$lib/types';

	type ReadFileMeta = {
		fileName: string;
		lineRange: { start: number; end: number } | null;
		language: string;
	};

	function parseReadFileMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined
	): ReadFileMeta | null {
		if (toolName !== 'read_file' || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const rawPath = args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(/[\\/]/).pop() || rawPath;

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
		const language = fileType ? fileType.replace(/^text:/, '') : 'text';

		return { fileName, lineRange, language };
	}

	type EditFileChange = {
		mode: 'replace' | 'delete' | 'append';
		lineStart: number;
		lineEnd: number;
		content: string;
	};

	type EditFileMeta = {
		fileName: string;
		filePath: string;
		language: string;
		changes: EditFileChange[];
		resultMessage?: string;
		totalLines?: number;
		errorMessage?: string;
	};

	function parseEditFileMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined,
		toolResultString: string | undefined
	): EditFileMeta | null {
		if (toolName !== 'edit_file' || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const rawPath = args.input_path ?? args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(/[\\/]/).pop() || rawPath;
		const language = getFileTypeByExtension(rawPath)?.replace(/^text:/, '') ?? 'text';

		const rawChanges = Array.isArray(args.changes) ? args.changes : [];
		const changes: EditFileChange[] = [];
		for (const c of rawChanges) {
			if (!c || typeof c !== 'object' || Array.isArray(c)) continue;
			const obj = c as Record<string, unknown>;
			const modeRaw = typeof obj.mode === 'string' ? obj.mode : '';
			if (modeRaw !== 'replace' && modeRaw !== 'delete' && modeRaw !== 'append') continue;
			changes.push({
				mode: modeRaw,
				lineStart: Number(obj.line_start) || 0,
				lineEnd: Number(obj.line_end) || 0,
				content: typeof obj.content === 'string' ? obj.content : ''
			});
		}

		let resultMessage: string | undefined;
		let totalLines: number | undefined;
		let errorMessage: string | undefined;
		if (toolResultString) {
			try {
				const parsed: unknown = JSON.parse(toolResultString);
				if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
					const obj = parsed as Record<string, unknown>;
					if (typeof obj.error === 'string') {
						errorMessage = obj.error;
					} else {
						if (typeof obj.result === 'string') {
							resultMessage = obj.result;
						}
						if (Number.isFinite(Number(obj.lines))) {
							totalLines = Number(obj.lines);
						}
					}
				}
			} catch {
				// Plain string tool result = legacy success, no rich metadata.
			}
		}

		return { fileName, filePath: rawPath, language, changes, resultMessage, totalLines, errorMessage };
	}

	function describeEditChange(change: EditFileChange): string {
		if (change.mode === 'append') {
			if (change.lineStart === -1) return 'append at end of file';
			return `append after line ${change.lineEnd}`;
		}
		if (change.mode === 'delete') {
			if (change.lineStart === change.lineEnd) return `delete line ${change.lineStart}`;
			return `delete lines ${change.lineStart}-${change.lineEnd}`;
		}
		if (change.lineStart === change.lineEnd) {
			return `replace line ${change.lineStart}`;
		}
		return `replace lines ${change.lineStart}-${change.lineEnd}`;
	}

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
		if (toolName !== 'write_file' || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const rawPath = args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(/[\\/]/).pop() || rawPath;
		const content = typeof args.content === 'string' ? args.content : '';
		const language = getFileTypeByExtension(rawPath)?.replace(/^text:/, '') ?? 'text';

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

	type ExecShellCommandMeta = {
		command: string;
	};

	function parseExecShellCommandMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined
	): ExecShellCommandMeta | null {
		if (toolName !== 'exec_shell_command' || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const commandRaw = args.command ?? args.cmd ?? args.shell_command;
		if (typeof commandRaw !== 'string' || !commandRaw) return null;

		return { command: commandRaw };
	}

	function parseExecShellCommandError(toolResultString: string | undefined): string | undefined {
		if (!toolResultString) return undefined;
		try {
			const parsed: unknown = JSON.parse(toolResultString);
			if (
				parsed &&
				typeof parsed === 'object' &&
				!Array.isArray(parsed) &&
				typeof (parsed as Record<string, unknown>).error === 'string'
			) {
				return (parsed as { error: string }).error;
			}
		} catch {
			// Plain-text result = stdout/stderr, no structured error to surface.
		}
		return undefined;
	}

	interface Props {
		section: AgenticSection;
		attachments?: DatabaseMessageExtra[];
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { section, attachments, open, isStreaming, onToggle }: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming));
	const toolUi = $derived(getBuiltinToolUi(section.toolName));
	const toolIcon = $derived(showSpinner ? Loader2 : (toolUi?.icon ?? Wrench));
	const toolIconClass = $derived(showSpinner ? 'h-4 w-4 animate-spin' : 'h-4 w-4');

	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithImages(section.toolResult, attachments) : []
	);

	const readFileMeta = $derived(parseReadFileMeta(section.toolName, section.toolArgs));
	const editFileMeta = $derived(
		parseEditFileMeta(section.toolName, section.toolArgs, section.toolResult)
	);
	const writeFileMeta = $derived(
		parseWriteFileMeta(section.toolName, section.toolArgs, section.toolResult)
	);
	const execShellMeta = $derived(parseExecShellCommandMeta(section.toolName, section.toolArgs));
	const execShellError = $derived(parseExecShellCommandError(section.toolResult));

	const hasCustomTitle = $derived(
		readFileMeta !== null ||
			writeFileMeta !== null ||
			editFileMeta !== null ||
			execShellMeta !== null
	);
	const title = $derived(hasCustomTitle ? '' : toolUi?.label ?? section.toolName ?? '');

	const subtitle = $derived(
		showSpinner
			? 'executing...'
			: editFileMeta?.errorMessage || writeFileMeta?.errorMessage || execShellError
				? 'failed'
				: isStreamingCall && !isStreaming
					? 'incomplete'
					: undefined
	);
</script>

{#snippet readFileTitle()}
	<span class="text-muted-foreground">Read file </span>
	<span class="font-mono">{readFileMeta?.fileName}</span>
	{#if readFileMeta?.lineRange}
		<span class="text-muted-foreground"
			>{' '}(lines {readFileMeta.lineRange.start}-{readFileMeta.lineRange.end})</span
		>
	{/if}
{/snippet}

{#snippet writeFileTitle()}
	<span class="text-muted-foreground">Write file </span>
	<span class="font-mono">{writeFileMeta?.filePath}</span>
	{#if writeFileMeta?.errorMessage}
		<span class="ml-1 text-xs italic text-muted-foreground/70">(failed)</span>
	{/if}
{/snippet}

{#snippet editFileTitle()}
	<span class="text-muted-foreground">Edit file </span>
	<span class="font-mono">{editFileMeta?.filePath}</span>
	{#if editFileMeta?.errorMessage}
		<span class="ml-1 text-xs italic text-muted-foreground/70">(failed)</span>
	{/if}
{/snippet}

{#snippet execShellTitle()}
	<span class="font-mono">{execShellMeta?.command}</span>
{/snippet}

<CollapsibleContentBlock
	{open}
	class="my-2"
	variant={execShellMeta ? 'terminal' : 'default'}
	icon={toolIcon}
	iconClass={toolIconClass}
	{title}
	titleSnippet={readFileMeta
		? readFileTitle
		: writeFileMeta
			? writeFileTitle
			: editFileMeta
				? editFileTitle
				: execShellMeta
					? execShellTitle
					: undefined}
	{subtitle}
	{isStreaming}
	{onToggle}
>
	{#if execShellMeta}
		{#if isPending}
			<div class="flex items-center gap-2 text-xs text-muted-foreground/70">
				<Loader2 class="h-3 w-3 animate-spin" />
				Running...
			</div>
		{:else if execShellError}
			<div class="flex items-start gap-2 text-xs text-red-600 italic dark:text-red-400">
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
				<span>{execShellError}</span>
			</div>
		{:else if section.toolResult}
			<div class="max-h-96 overflow-auto">
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
	{:else if isStreamingCall}
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
				maxHeight="22rem"
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
	{:else if writeFileMeta}
		{#if isPending}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for writer...
			</div>
		{:else if writeFileMeta.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
			>
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
				<span>{writeFileMeta.errorMessage}</span>
			</div>
		{:else}
			<SyntaxHighlightedCode
				code={writeFileMeta.content}
				language={writeFileMeta.language}
				maxHeight="22rem"
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
	{:else if readFileMeta}
		{#if section.toolResult}
			<SyntaxHighlightedCode
				code={section.toolResult}
				language={readFileMeta.language}
				maxHeight="22rem"
			/>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for file content...
			</div>
		{/if}
	{:else if editFileMeta}
		{#if isPending}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for editor...
			</div>
		{:else if editFileMeta.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
			>
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
				<span>{editFileMeta.errorMessage}</span>
			</div>
		{:else if editFileMeta.changes.length > 0}
			{#each editFileMeta.changes as change, ci (ci)}
				<div class={ci === 0 ? '' : 'mt-3'}>
					<div class="mb-1.5 text-xs text-muted-foreground/70 italic">
						{describeEditChange(change)}
					</div>
					{#if change.mode !== 'delete' && change.content}
						<SyntaxHighlightedCode
							code={change.content}
							language={editFileMeta.language}
							maxHeight="22rem"
						/>
					{/if}
				</div>
			{/each}
			<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
				{#if editFileMeta.resultMessage}
					{editFileMeta.resultMessage}{editFileMeta.totalLines != null
						? '\u00A0\u00B7\u00A0'
						: ''}{/if}
				{#if editFileMeta.totalLines != null}
					<span class="font-mono">{editFileMeta.totalLines}</span>
					lines
				{/if}
			</div>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				No changes
			</div>
		{/if}
	{:else}
		<div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
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
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				No output
			</div>
		{/if}
	{/if}
</CollapsibleContentBlock>
