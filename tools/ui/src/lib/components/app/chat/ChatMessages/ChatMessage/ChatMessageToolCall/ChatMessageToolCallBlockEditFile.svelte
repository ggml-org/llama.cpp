<script lang="ts">
	import { Loader2, Wrench, XCircle } from '@lucide/svelte';
	import { CollapsibleContentBlock } from '$lib/components/app';
	import { AgenticSectionType, BuiltInTool } from '$lib/enums';
	import { FILE_PATH_SEPARATOR_REGEX } from '$lib/constants';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { AgenticSection } from '$lib/utils';
	import { parsePartialJsonArgs } from './parse-partial-json-args';
	import { computeLineDiff, type DiffLine } from './compute-line-diff';

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
	const toolIconClass = $derived(showSpinner ? 'h-4 w-4 animate-spin' : 'h-4 w-4');
	const mcpServerFavicon = $derived(mcpStore.getServerFaviconForTool(section.toolName));
	const iconUrl = $derived(
		!showSpinner && !toolUi?.icon && mcpServerFavicon ? mcpServerFavicon : null
	);

	type EditFileEdit = {
		oldText: string;
		newText: string;
	};

	type EditFileMeta = {
		fileName: string;
		filePath: string;
		edits: EditFileEdit[];
		resultMessage?: string;
		editsApplied?: number;
		errorMessage?: string;
	};

	function parseEditFileMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined,
		toolResultString: string | undefined
	): EditFileMeta | null {
		if (toolName !== BuiltInTool.EDIT_FILE || !toolArgsString) return null;

		const args = parsePartialJsonArgs(toolArgsString);
		if (!args) return null;

		const rawPath = args.path ?? args.file_path ?? args.filePath;
		if (typeof rawPath !== 'string' || !rawPath) return null;

		const fileName = rawPath.split(FILE_PATH_SEPARATOR_REGEX).pop() || rawPath;

		const rawEdits = Array.isArray(args.edits) ? args.edits : [];
		const edits: EditFileEdit[] = [];
		for (const e of rawEdits) {
			if (!e || typeof e !== 'object' || Array.isArray(e)) continue;
			const obj = e as Record<string, unknown>;
			const oldText = typeof obj.old_text === 'string' ? obj.old_text : '';
			if (!oldText) continue;
			const newText = typeof obj.new_text === 'string' ? obj.new_text : '';
			edits.push({ oldText, newText });
		}

		let resultMessage: string | undefined;
		let editsApplied: number | undefined;
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
						if (Number.isFinite(Number(obj.edits_applied))) {
							editsApplied = Number(obj.edits_applied);
						}
					}
				}
			} catch {
				// Plain string tool result = legacy success, no rich metadata.
			}
		}

		return {
			fileName,
			filePath: rawPath,
			edits,
			resultMessage,
			editsApplied,
			errorMessage
		};
	}

	const editFileMeta = $derived(
		parseEditFileMeta(section.toolName, section.toolArgs, section.toolResult)
	);

	// Structured per-line diff for each edit. Rendered directly (instead of
	// through SyntaxHighlightedCode) so we can attach a line-number gutter
	// alongside each diff row.
	const editDiffs = $derived(
		(editFileMeta?.edits ?? []).map((edit) => computeLineDiff(edit.oldText, edit.newText))
	);

	const subtitle = $derived(
		showSpinner
			? 'executing...'
			: editFileMeta?.errorMessage
				? 'failed'
				: isStreamingCall && !isStreaming
					? 'incomplete'
					: undefined
	);

	function markerFor(kind: DiffLine['kind']): string {
		if (kind === 'add') return '+';
		if (kind === 'remove') return '-';
		return ' ';
	}
</script>

{#snippet editFileTitle()}
	<span class="text-muted-foreground">Edit file </span>
	<span class="font-mono">{editFileMeta?.filePath}</span>
	{#if editFileMeta?.errorMessage}
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
	titleSnippet={editFileTitle}
	{subtitle}
	{onToggle}
>
	{#if editFileMeta?.errorMessage}
		<div
			class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
		>
			<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
			<span>{editFileMeta.errorMessage}</span>
		</div>
	{:else if editFileMeta && editFileMeta.edits.length > 0}
		{#each editDiffs as diffLines, ei (ei)}
			<div class={ei === 0 ? '' : 'mt-3'}>
				<div class="mb-1.5 text-xs text-muted-foreground/70 italic">
					Edit {ei + 1}&nbsp;of&nbsp;{editFileMeta?.edits.length ?? 0}
				</div>
				<div class="diff-block">
					<div class="diff-pre">
						{#each diffLines as line, li (li)}
							<div class="diff-line diff-{line.kind}">
								<span class="diff-old-num">{line.oldLine ?? ''}</span>
								<span class="diff-marker">{markerFor(line.kind)}</span>
								<span class="diff-new-num">{line.newLine ?? ''}</span>
								<span class="diff-text">{line.text || '\u00A0'}</span>
							</div>
						{/each}
					</div>
				</div>
			</div>
		{/each}
		<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
			{#if editFileMeta.resultMessage}
				{editFileMeta.resultMessage}{editFileMeta.editsApplied != null
					? '\u00A0\u00B7\u00A0'
					: ''}{/if}
			{#if editFileMeta.editsApplied != null}
				<span class="font-mono">{editFileMeta.editsApplied}</span>
				{editFileMeta.editsApplied === 1 ? 'edit' : 'edits'}&nbsp;applied
			{/if}
		</div>
	{:else}
		<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No edits</div>
	{/if}
</CollapsibleContentBlock>

<style>
	.diff-block {
		max-height: 22rem;
		overflow: auto;
		border-radius: 0.75rem;
		border-width: 1px;
		border-color: color-mix(in oklch, var(--border) 30%, transparent);
		background: var(--code-background);
		box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
	}

	:global(.dark) .diff-block {
		border-color: color-mix(in oklch, var(--border) 20%, transparent);
	}

	/* Each row is a 4-column grid: old-line#, marker, new-line#, text.
	 * The gutters stay fixed-width so the text column lines up unversally. */
	.diff-line {
		display: grid;
		grid-template-columns: 3.25rem 1.5rem 3.25rem 1fr;
		font-family: var(--font-mono);
		font-size: 11px;
		line-height: 1.65;
		align-items: stretch;
	}

	.diff-old-num,
	.diff-new-num {
		text-align: right;
		padding-right: 0.5rem;
		user-select: none;
		color: color-mix(in oklch, var(--muted-foreground) 70%, transparent);
		font-variant-numeric: tabular-nums;
	}

	.diff-marker {
		text-align: center;
		color: color-mix(in oklch, var(--muted-foreground) 70%, transparent);
		user-select: none;
	}

	.diff-line.diff-add {
		background-color: #f0fff4;
		color: #22863a;
	}
	.diff-line.diff-add .diff-new-num,
	.diff-line.diff-add .diff-marker {
		color: #22863a;
	}

	.diff-line.diff-remove {
		background-color: #ffeef0;
		color: #b31d28;
	}
	.diff-line.diff-remove .diff-old-num,
	.diff-line.diff-remove .diff-marker {
		color: #b31d28;
	}

	.diff-line.diff-add .diff-old-num,
	.diff-line.diff-remove .diff-new-num {
		/* Empty gutter columns for add/remove rows mirror git unification
		 * (added lines don't have an old number, removed lines don't have a
		 * new number). Keep them visible so columns stay aligned across
		 * mixed rows. */
		opacity: 0;
	}

	.diff-text {
		padding-left: 0.4rem;
		padding-right: 0.5rem;
		white-space: pre;
		overflow-x: auto;
		min-width: 0;
	}

	:global(.dark) .diff-line.diff-add {
		background-color: #033a16;
		color: #aff5b4;
	}
	:global(.dark) .diff-line.diff-add .diff-new-num,
	:global(.dark) .diff-line.diff-add .diff-marker {
		color: #aff5b4;
	}
	:global(.dark) .diff-line.diff-remove {
		background-color: #67060c;
		color: #ffdcd7;
	}
	:global(.dark) .diff-line.diff-remove .diff-old-num,
	:global(.dark) .diff-line.diff-remove .diff-marker {
		color: #ffdcd7;
	}
</style>
