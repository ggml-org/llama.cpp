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

	type GrepSearchMatch = {
		file: string;
		line?: number;
		content: string;
	};

	type GrepSearchMeta = {
		path: string;
		pattern: string;
		include: string;
		exclude?: string;
		showLineNumbers: boolean;
		matches: GrepSearchMatch[];
		totalMatches?: number;
		errorMessage?: string;
	};

	function parseGrepSearchMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined,
		toolResultString: string | undefined
	): GrepSearchMeta | null {
		if (toolName !== BuiltInTool.GREP_SEARCH || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const path = typeof args.path === 'string' ? args.path : '';
		const pattern = typeof args.pattern === 'string' ? args.pattern : '';
		if (!path || !pattern) return null;

		const include = typeof args.include === 'string' && args.include ? args.include : '**';
		const exclude = typeof args.exclude === 'string' && args.exclude ? args.exclude : undefined;
		const showLineNumbers = args.return_line_numbers === true;

		let matches: GrepSearchMatch[] = [];
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
						const split = splitSearchSummaryList(
							obj.plain_text_response,
							(total) => (totalMatches = total)
						);
						matches = split.lines.map((line) => parseGrepLine(line, showLineNumbers));
					}
				}
			} catch {
				const split = splitSearchSummaryList(toolResultString, (total) => (totalMatches = total));
				matches = split.lines.map((line) => parseGrepLine(line, showLineNumbers));
			}
		}

		return {
			path,
			pattern,
			include,
			exclude,
			showLineNumbers,
			matches,
			totalMatches,
			errorMessage
		};
	}

	function parseGrepLine(line: string, showLineNumbers: boolean): GrepSearchMatch {
		// Server output: <file>:<content>            when return_line_numbers=false
		//               <file>:<lineno>:<content>   when return_line_numbers=true
		const firstColon = line.indexOf(':');
		if (firstColon === -1) {
			return { file: line, content: '' };
		}
		const file = line.slice(0, firstColon);
		const tail = line.slice(firstColon + 1);

		if (!showLineNumbers) {
			return { file, content: tail };
		}

		const secondColon = tail.indexOf(':');
		if (secondColon === -1) {
			return { file, content: tail };
		}
		const lineNum = parseInt(tail.slice(0, secondColon), 10);
		return {
			file,
			line: Number.isFinite(lineNum) ? lineNum : undefined,
			content: tail.slice(secondColon + 1)
		};
	}

	const grepMeta = $derived(
		parseGrepSearchMeta(section.toolName, section.toolArgs, section.toolResult)
	);
</script>

{#snippet grepSearchTitle()}
	{#if grepMeta}
		<span class="text-muted-foreground">Search for&nbsp;</span>
		<span class="font-mono">{grepMeta.pattern}</span>
		<span class="text-muted-foreground">&nbsp;in&nbsp;</span>
		<span class="font-mono">{grepMeta.path}</span>
	{/if}
{/snippet}

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	title=""
	titleSnippet={grepSearchTitle}
	{onToggle}
>
	{#if isPending}
		<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">Searching...</div>
	{:else if grepMeta?.errorMessage}
		<div
			class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
		>
			<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
			<span>{grepMeta.errorMessage}</span>
		</div>
	{:else if grepMeta && grepMeta.matches.length > 0}
		<div class="max-h-96 overflow-auto">
			{#each grepMeta.matches as match, mi (mi)}
				<div class="font-mono text-[11px] leading-relaxed">
					<span class="text-muted-foreground/70">{match.file}</span>
					{#if grepMeta.showLineNumbers && match.line != null}
						<span class="text-muted-foreground/70">:{match.line}</span>
					{/if}
					<span class="text-muted-foreground/70">:</span>
					<span>{match.content}</span>
				</div>
			{/each}
		</div>
		<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
			Total matches: <span class="font-mono"
				>{grepMeta.totalMatches ?? grepMeta.matches.length}</span
			>
			{#if grepMeta.showLineNumbers}
				&nbsp;<span class="italic">(with line numbers)</span>
			{/if}
		</div>
	{:else}
		<div class="text-xs text-muted-foreground/70 italic">No matches</div>
		<div class="mt-1.5 text-xs text-muted-foreground/70 italic">
			Total matches: <span class="font-mono">{grepMeta?.totalMatches ?? 0}</span>
		</div>
	{/if}
</CollapsibleContentBlock>
