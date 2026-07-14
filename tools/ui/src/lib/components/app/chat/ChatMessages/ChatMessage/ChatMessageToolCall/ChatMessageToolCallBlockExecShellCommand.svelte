<script lang="ts">
	import { Loader2, Wrench, XCircle } from '@lucide/svelte';
	import { CollapsibleTerminalBlock } from '$lib/components/app';
	import { AgenticSectionType, BuiltInTool } from '$lib/enums';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { parseToolResultWithImages, type AgenticSection, type ToolResultLine } from '$lib/utils';
	import type { DatabaseMessageExtra } from '$lib/types';
	import { parseExecShellCommandError } from './parse-exec-shell-error';

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
	const toolUi = $derived(getBuiltinToolUi(section.toolName));
	const toolIcon = $derived(showSpinner ? Loader2 : (toolUi?.icon ?? Wrench));
	const toolIconClass = $derived(showSpinner ? 'h-4 w-4 animate-spin' : 'h-4 w-4');
	const mcpServerFavicon = $derived(mcpStore.getServerFaviconForTool(section.toolName));
	const iconUrl = $derived(
		!showSpinner && !toolUi?.icon && mcpServerFavicon ? mcpServerFavicon : null
	);

	type ExecShellCommandMeta = {
		command: string;
	};

	function parseExecShellCommandMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined
	): ExecShellCommandMeta | null {
		if (toolName !== BuiltInTool.EXEC_SHELL_COMMAND || !toolArgsString) return null;

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

	const execShellMeta = $derived(parseExecShellCommandMeta(section.toolName, section.toolArgs));
	const execShellError = $derived(parseExecShellCommandError(section.toolResult));

	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithImages(section.toolResult, attachments) : []
	);

	const subtitle = $derived(
		showSpinner
			? 'executing...'
			: execShellError
				? 'failed'
				: isStreamingCall && !isStreaming
					? 'incomplete'
					: undefined
	);
</script>

{#snippet execShellTitle()}
	<span class="font-mono">{execShellMeta?.command}</span>
{/snippet}

<CollapsibleTerminalBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	title=""
	titleSnippet={execShellTitle}
	{subtitle}
	{onToggle}
>
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
</CollapsibleTerminalBlock>
