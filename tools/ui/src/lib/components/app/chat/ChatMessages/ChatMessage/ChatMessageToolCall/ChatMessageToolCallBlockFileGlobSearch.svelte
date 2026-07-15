<script lang="ts">
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SPIN } from '$lib/constants/css-classes';
	import { Loader2, Wrench, XCircle } from '@lucide/svelte';
	import { CollapsibleContentBlock } from '$lib/components/app';
	import { AgenticSectionType, BuiltInTool } from '$lib/enums';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { splitSearchSummaryList, type AgenticSection } from '$lib/utils';

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

	type FileGlobSearchMeta = {
		path: string;
		include: string;
		exclude?: string;
		matches: string[];
		totalMatches?: number;
		errorMessage?: string;
	};

	function splitGlobMatches(text: string, captureTotal: (n: number) => void): string[] {
		return splitSearchSummaryList(text, captureTotal).lines;
	}

	function parseFileGlobSearchMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined,
		toolResultString: string | undefined
	): FileGlobSearchMeta | null {
		if (toolName !== BuiltInTool.FILE_GLOB_SEARCH || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const path = typeof args.path === 'string' ? args.path : '';
		const include = typeof args.include === 'string' && args.include ? args.include : '**';
		const exclude = typeof args.exclude === 'string' && args.exclude ? args.exclude : undefined;
		if (!path) return null;

		let matches: string[] = [];
		let totalMatches: number | undefined;
		let errorMessage: string | undefined;

		if (toolResultString) {
			try {
				const parsed: unknown = JSON.parse(toolResultString);
				if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
					const obj = parsed as Record<string, unknown>;
					if (typeof obj.error === 'string') {
						errorMessage = obj.error;
					} else if (typeof obj.plain_text_response === 'string') {
						matches = splitGlobMatches(obj.plain_text_response, (total) => (totalMatches = total));
					}
				}
			} catch {
				matches = splitGlobMatches(toolResultString, (total) => (totalMatches = total));
			}
		}

		return { path, include, exclude, matches, totalMatches, errorMessage };
	}

	const fileGlobMeta = $derived(
		parseFileGlobSearchMeta(section.toolName, section.toolArgs, section.toolResult)
	);
</script>

{#snippet fileGlobTitle()}
	{#if fileGlobMeta}
		<span class="text-muted-foreground"
			>{fileGlobMeta.include === '**' ? 'List files' : 'Search files'}&nbsp;</span
		>
		{#if fileGlobMeta.include !== '**'}
			<span class="font-mono">{fileGlobMeta.include}</span>
		{/if}
		<span class="text-muted-foreground">&nbsp;in&nbsp;</span>
		<span class="font-mono">{fileGlobMeta.path}</span>
	{/if}
{/snippet}

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	title=""
	titleSnippet={fileGlobTitle}
	{onToggle}
>
	{#if isPending}
		<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">Searching...</div>
	{:else if fileGlobMeta?.errorMessage}
		<div
			class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
		>
			<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
			<span>{fileGlobMeta.errorMessage}</span>
		</div>
	{:else if fileGlobMeta && fileGlobMeta.matches.length > 0}
		<div class="max-h-96 overflow-auto">
			{#each fileGlobMeta.matches as match, i (i)}
				<div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{match}</div>
			{/each}
		</div>
		<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
			Total matches: <span class="font-mono"
				>{fileGlobMeta.totalMatches ?? fileGlobMeta.matches.length}</span
			>
		</div>
	{:else}
		<div class="text-xs text-muted-foreground/70 italic">No matches</div>
		<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
			Total matches: <span class="font-mono">{fileGlobMeta?.totalMatches ?? 0}</span>
		</div>
	{/if}
</CollapsibleContentBlock>
