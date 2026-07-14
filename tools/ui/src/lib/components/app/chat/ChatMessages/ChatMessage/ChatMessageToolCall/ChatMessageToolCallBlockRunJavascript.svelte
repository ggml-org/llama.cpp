<script lang="ts">
	import { Loader2, Wrench, XCircle, Terminal } from '@lucide/svelte';
	import { CollapsibleContentBlock, SyntaxHighlightedCode } from '$lib/components/app';
	import { AgenticSectionType, BuiltInTool, FileTypeText } from '$lib/enums';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { type AgenticSection } from '$lib/utils';

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
	const toolIconClass = $derived(showSpinner ? 'h-4 w-4 animate-spin' : 'h-4 w-4');
	const mcpServerFavicon = $derived(mcpStore.getServerFaviconForTool(section.toolName));
	const iconUrl = $derived(
		!showSpinner && !toolUi?.icon && mcpServerFavicon ? mcpServerFavicon : null
	);

	type RunJavascriptMeta = {
		code: string;
		timeoutMs?: number;
		errorMessage?: string;
	};

	function parseRunJavascriptMeta(
		toolName: string | undefined,
		toolArgsString: string | undefined,
		toolResultString: string | undefined
	): RunJavascriptMeta | null {
		if (toolName !== BuiltInTool.RUN_JAVASCRIPT || !toolArgsString) return null;

		let args: Record<string, unknown>;
		try {
			const parsed: unknown = JSON.parse(toolArgsString);
			if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
			args = parsed as Record<string, unknown>;
		} catch {
			return null;
		}

		const code = typeof args.code === 'string' ? args.code : '';
		if (!code) return null;

		const timeoutRaw = Number(args.timeout_ms);
		const timeoutMs = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? timeoutRaw : undefined;

		let errorMessage: string | undefined;
		if (toolResultString) {
			try {
				const parsed: unknown = JSON.parse(toolResultString);
				if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
					const obj = parsed as Record<string, unknown>;
					if (typeof obj.error === 'string') errorMessage = obj.error;
				}
			} catch {
				// SandboxService.formatReply emits 'Error: <msg>' on failure.
				const errorLine = toolResultString
					.split('\n')
					.map((line) => line.trim())
					.find((line) => line.startsWith('Error:'));
				if (errorLine) errorMessage = errorLine.slice('Error:'.length).trim();
			}
		}

		return { code, timeoutMs, errorMessage };
	}

	const runJsMeta = $derived(
		parseRunJavascriptMeta(section.toolName, section.toolArgs, section.toolResult)
	);
</script>

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	title={toolUi?.label ?? section.toolName ?? ''}
	{onToggle}
>
	{#if isPending}
		<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">Running...</div>
	{:else if runJsMeta?.errorMessage}
		<div
			class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
		>
			<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
			<span>{runJsMeta.errorMessage}</span>
		</div>
		<div class="mt-3">
			<SyntaxHighlightedCode
				code={runJsMeta.code}
				language={FileTypeText.JAVASCRIPT}
				maxHeight="22rem"
				streaming={isCodeStreaming}
			/>
		</div>
	{:else if runJsMeta}
		<SyntaxHighlightedCode
			code={runJsMeta.code}
			language={FileTypeText.JAVASCRIPT}
			maxHeight="22rem"
			streaming={isCodeStreaming}
		/>
		<div class="mb-2 mt-3 flex items-center gap-2 text-xs text-muted-foreground/70">
			<Terminal class="h-3 w-3" />
			<span>Console</span>
			{#if runJsMeta.timeoutMs != null}
				<span class="font-mono">&middot;&nbsp;timeout&nbsp;{runJsMeta.timeoutMs}&nbsp;ms</span>
			{/if}
		</div>
		{#if section.toolResult}
			<div class="mt-1">
				<SyntaxHighlightedCode
					code={section.toolResult}
					language={FileTypeText.JAVASCRIPT}
					maxHeight="22rem"
				/>
			</div>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">No output</div>
		{/if}
	{/if}
</CollapsibleContentBlock>
