<script lang="ts">
	// Block for `exec_shell_command`. Collapsed, it reads as one line: "Ran"
	// or the model's description of the call, always followed by the command.
	// Open, it shows the terminal: the prompt line and the output. It treats
	// "live" output chunks as active even after the call resolved, so the
	// spinner stays on while stdout is still flowing.
	// The scroll-to-bottom auto-scroll logic mirrors what was here
	// before extraction.

	import { parseExecShellCommandMeta } from './parsers/exec-shell-command';
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { AlertTriangle, Check, CircleStop, Loader2, XCircle } from '@lucide/svelte';
	import {
		SETTINGS_KEYS,
		TOOL_RUNTIME_SCROLL_AT_BOTTOM_THRESHOLD_PX
	} from '#lib/constants/index.ts';
	import { AgenticSectionType, AttachmentType } from '#lib/enums/index.ts';
	import { settingsStore, toolsStore } from '#lib/stores/index.ts';
	import type { AgenticSection, DatabaseMessageExtra, ToolResultLine } from '#lib/types/index.ts';
	import {
		abbreviateHome,
		type ExecShellExitStatus,
		highlightCode,
		isExitCodeSummaryLine,
		parseExecShellCommandError,
		parseExecShellCommandExitStatus,
		parseToolResultWithMedia
	} from '#lib/utils/index.ts';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		/** True while the agentic loop is streaming output chunks for THIS
		 *  tool call. Drives max-height + auto-scroll while true; releases
		 *  them when the loop reports this call as done. */
		isExecuting?: boolean;
		/** Working directory of the conversation this call belongs to; stands
		 *  in for `section.toolCwd` until the result message persists the
		 *  real one. */
		conversationCwd?: string;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let {
		attachments,
		conversationCwd,
		isExecuting = false,
		isStreaming,
		onToggle,
		open,
		section
	}: Props = $props();

	// `isLive` covers all in-flight phases: pre-chunk spinner and
	// streaming itself. Frozen output (tool done while agent continues)
	// is not live.
	const isLive = $derived(isExecuting);

	const isPreResult = $derived(
		section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_STREAMING
	);

	// The model is still typing the call while the args JSON doesn't parse;
	// the title keeps a terminal cursor until it does.
	const commandStreaming = $derived(
		isStreaming && isPreResult && !isParsableJson(section.toolArgs)
	);

	const execShellMeta = $derived(parseExecShellCommandMeta(section));
	const execShellError = $derived(parseExecShellCommandError(section.toolResult));
	const execShellExitStatus: ExecShellExitStatus | undefined = $derived(
		parseExecShellCommandExitStatus(section.toolResult)
	);

	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithMedia(section.toolResult, attachments) : []
	);

	// Drop the trailing "[exit code: N]" line - rendered as a colored
	// badge below. During streaming we keep it so a partial stream still
	// shows the status once the final chunk lands.
	const outputLines: ToolResultLine[] = $derived(
		execShellExitStatus && parsedLines.length > 0
			? parsedLines.slice(0, parsedLines.length - 1)
			: parsedLines
	);

	const isExitCodeFinalLine = $derived(
		execShellExitStatus !== undefined &&
			parsedLines.length > 0 &&
			isExitCodeSummaryLine(parsedLines[parsedLines.length - 1].text, execShellExitStatus)
	);

	// Highlight just the command for the title; the (typically large)
	// output blob uses bare monospace to skip hljs per-line highlighting.
	const highlightedCommandHtml = $derived(
		execShellMeta ? highlightCode(execShellMeta.command, 'bash') : ''
	);

	// The working directory the command ran with, persisted per call on the
	// tool result message (it travels via the x-tool-cwd header, not the tool
	// args). Until that message exists, the conversation cwd is the best known
	// value - it is what the agentic loop passes when executing the call.
	const cwd = $derived(
		section.toolCwd ?? (isStreaming && isPreResult ? conversationCwd : undefined)
	);
	const home = $derived(toolsStore.serverHome);
	const wdDisplay = $derived(abbreviateHome(cwd ?? '', home));

	const exitBadgeClass = $derived(
		execShellExitStatus?.timedOut
			? 'exit-badge warning'
			: execShellExitStatus?.code === 0
				? 'exit-badge success'
				: 'exit-badge failure'
	);

	const useFullHeightCodeBlocks = $derived(
		Boolean(settingsStore.config[SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS])
	);

	const autoScroll = $derived(isLive && !useFullHeightCodeBlocks);

	const SCROLL_BOTTOM_THRESHOLD_PX = TOOL_RUNTIME_SCROLL_AT_BOTTOM_THRESHOLD_PX;

	let scrollEl: HTMLDivElement | undefined = $state();
	let userScrolledUp = $state(false);
	let lastScrollTop = 0;
	let pendingFrame: number | null = null;

	function isAtBottom(): boolean {
		if (!scrollEl) return false;

		return (
			scrollEl.scrollHeight - scrollEl.clientHeight - scrollEl.scrollTop <=
			SCROLL_BOTTOM_THRESHOLD_PX
		);
	}

	function isParsableJson(value: string | undefined): boolean {
		if (!value) return false;

		try {
			JSON.parse(value);

			return true;
		} catch {
			return false;
		}
	}

	function scrollToBottomOnFrame() {
		if (pendingFrame !== null || !scrollEl || userScrolledUp) return;

		pendingFrame = requestAnimationFrame(() => {
			pendingFrame = null;

			// Re-check on rAF - user may scroll between scheduling and paint.
			if (scrollEl && !userScrolledUp) {
				scrollEl.scrollTop = scrollEl.scrollHeight;
			}
		});
	}

	function handleScrollEvent() {
		if (!scrollEl) return;

		const isScrollingUp = scrollEl.scrollTop < lastScrollTop;

		if (isScrollingUp && !isAtBottom()) {
			userScrolledUp = true;
		} else if (isAtBottom()) {
			userScrolledUp = false;
		}

		lastScrollTop = scrollEl.scrollTop;
	}

	$effect(() => {
		void section.toolResult;

		if (!scrollEl || !autoScroll) return;

		scrollToBottomOnFrame();
	});

	$effect(() => {
		// Catch layout changes that don't touch toolResult (line-wrap
		// reflow, image attaches, hljs settle).
		if (!scrollEl || !autoScroll) return;

		const observer = new MutationObserver(() => scrollToBottomOnFrame());

		observer.observe(scrollEl, {
			characterData: true,
			childList: true,
			subtree: true
		});

		return () => observer.disconnect();
	});

	$effect(() => {
		// Reset on stream end so the next render (full-height) starts
		// pinned.
		if (!isLive) {
			userScrolledUp = false;
			lastScrollTop = 0;
		}
	});
</script>

{#snippet promptLine()}
	{#if cwd}
		<span class="exec-wd" title={cwd}>{wdDisplay}</span>
	{/if}

	<span class="exec-prompt">$</span>

	{#if highlightedCommandHtml}
		<span class="font-mono">{@html highlightedCommandHtml}</span>
	{:else}
		<span class="font-mono">{execShellMeta?.command}</span>
	{/if}

	{#if commandStreaming}
		<span class="exec-cursor"></span>
	{/if}
{/snippet}

<ToolCallBlock
	emptyResultIsInterrupted
	extraLiveStreaming={isLive}
	{isStreaming}
	meta={execShellMeta ? { errorMessage: execShellError } : null}
	{onToggle}
	{open}
	{section}
>
	{#snippet titleSnippet(phrase)}
		<!-- The command is always in the title: a description is the model's
		     wording, and the reader approves and audits what actually runs.
		     Both take their own width while the row has room. When it runs out
		     the command gives way first, since the full command is one click away
		     and a cut description reads as noise. -->
		<span class="flex min-w-0 items-baseline gap-1.5">
			<span
				class={phrase.described ? 'min-w-0 truncate' : 'shrink-0'}
				title={phrase.described ? phrase.label : undefined}>{phrase.label}</span
			>

			{#if execShellMeta?.command}
				<span
					class="exec-title-command min-w-[2ch] shrink-[1000] truncate font-mono"
					title={execShellMeta.command}
				>
					{execShellMeta.command}
				</span>
			{/if}
		</span>
	{/snippet}

	{#snippet children(_meta, ctx)}
		<div class="terminal-frame">
			<div class="exec-prompt-line font-mono text-caption">
				{@render promptLine()}
			</div>

			{#if (ctx.isPending && !commandStreaming) || (isLive && !section.toolResult)}
				<div class="flex items-start gap-2 text-caption text-muted-foreground/70">
					<Loader2 class="size-icon-xs animate-spin" />
					Executing command...
				</div>
			{:else if ctx.isInterrupted}
				<div class="flex items-start gap-2 text-caption text-muted-foreground/70 italic">
					<CircleStop class="mt-0.5 h-3 w-3 shrink-0" />

					<span>Command interrupted before it returned output</span>
				</div>
			{:else if execShellError}
				<div class="flex items-start gap-2 text-caption text-fg-danger italic">
					<XCircle class="mt-0.5 size-icon-xs shrink-0" />

					<span>{execShellError}</span>
				</div>
			{:else if section.toolResult}
				<div
					bind:this={scrollEl}
					class={['terminal-output', !useFullHeightCodeBlocks && 'is-clamped']}
					onscroll={handleScrollEvent}
				>
					{#each outputLines as line, i (i)}
						<div class="font-mono text-caption leading-relaxed whitespace-pre-wrap">
							{line.text}
						</div>

						{#if line.media?.type === AttachmentType.IMAGE}
							<img
								alt={line.media.name}
								class="mt-2 mb-2 h-auto max-w-full rounded-lg"
								loading="lazy"
								src={line.media.base64Url}
							/>
						{/if}
					{/each}

					{#if isExitCodeFinalLine && execShellExitStatus}
						<div class={exitBadgeClass}>
							{#if execShellExitStatus.timedOut}
								<AlertTriangle class="size-icon-xs" />

								<span>timed out</span>

								<span class="exit-sep">&middot;</span>

								<span>exit {execShellExitStatus.code}</span>
							{:else if execShellExitStatus.code === 0}
								<Check class="size-icon-xs" />

								<span>exit 0</span>
							{:else}
								<XCircle class="size-icon-xs" />

								<span>exit {execShellExitStatus.code}</span>
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</div>
	{/snippet}
</ToolCallBlock>

<style>
	:root {
		--exec-wd-margin: 0.4rem;
	}

	.exec-wd {
		font-family: var(--font-mono);
		color: var(--fg-muted);
		margin-right: var(--exec-wd-margin);
	}

	.exec-prompt {
		font-family: var(--font-mono);
		color: var(--fg-muted);
		opacity: 0.55;
		margin-right: var(--exec-wd-margin);
	}

	.exec-cursor {
		display: inline-block;
		width: 0.5em;
		height: 1.05em;
		margin-left: 0.15rem;
		vertical-align: text-bottom;
		background: var(--fg-default);
		animation: exec-cursor-blink 1.1s steps(1) infinite;
	}

	@keyframes exec-cursor-blink {
		50% {
			opacity: 0;
		}
	}

	.exec-title-command {
		color: var(--fg-muted);
		font-size: 0.92em;
	}

	.terminal-frame {
		display: grid;
		gap: 0.5rem;
		min-width: 0;
		padding: 0.625rem 0.75rem;
		border-radius: var(--radius-md, 0.375rem);
		background: var(--surface-code);
		border: 1px solid color-mix(in oklch, var(--line-default) 30%, transparent);
	}

	.exec-prompt-line {
		word-break: break-word;
	}

	.exec-prompt-line :global(.font-mono) {
		white-space: pre-wrap;
	}

	.terminal-output.is-clamped {
		max-height: 28rem;
		overflow-y: auto;
		scrollbar-gutter: stable;
		padding-right: 0.25rem;
	}

	.exit-badge {
		display: inline-flex;
		align-items: center;
		gap: 0.35rem;
		margin-top: 0.5rem;
		padding: 0.2rem 0.55rem;
		border-radius: 0.375rem;
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 500;
		letter-spacing: 0.01em;
		line-height: 1;
	}

	/* the status tokens, which follow the theme on their own */
	.exit-badge.success {
		background: var(--surface-success);
		color: var(--fg-success);
	}

	.exit-badge.failure {
		background: var(--surface-danger);
		color: var(--fg-danger);
	}

	.exit-badge.warning {
		background: var(--surface-warning);
		color: var(--fg-warning);
	}

	.exit-sep {
		opacity: 0.45;
	}
</style>
