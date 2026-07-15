<script lang="ts">
	// Block for `exec_shell_command`. Unlike the other tools, this
	// renderer uses CollapsibleTerminalBlock (terminal-style frame)
	// and treats "live" output chunks as active even after the call
	// resolved, so the spinner stays on while stdout is still flowing.
	// The scroll-to-bottom auto-scroll logic mirrors what was here
	// before extraction.

	import { Check, Loader2, XCircle, AlertTriangle } from '@lucide/svelte';
	import { CollapsibleTerminalBlock } from '$lib/components/app';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { config } from '$lib/stores/settings.svelte';
	import { TOOL_RUNTIME_SCROLL_AT_BOTTOM_THRESHOLD_PX } from '$lib/constants/auto-scroll';
	import {
		highlightCode,
		isExitCodeSummaryLine,
		parseExecShellCommandError,
		parseExecShellCommandExitStatus,
		parseToolResultWithImages,
		type AgenticSection,
		type ExecShellExitStatus,
		type ToolResultLine
	} from '$lib/utils';
	import { parseExecShellCommandMeta } from './parsers/exec-shell-command';
	import type { DatabaseMessageExtra } from '$lib/types';
	import ToolCallBlock from './ToolCallBlock.svelte';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		/** True while the agentic loop is streaming output chunks for THIS
		 *  tool call. Drives max-height + auto-scroll while true; releases
		 *  them when the loop reports this call as done. */
		isExecuting?: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let { section, open, isStreaming, isExecuting = false, attachments, onToggle }: Props = $props();

	// `isLive` covers all in-flight phases: pre-chunk spinner and
	// streaming itself. Frozen output (tool done while agent continues)
	// is not live.
	const isLive = $derived(isExecuting);

	const execShellMeta = $derived(parseExecShellCommandMeta(section));
	const execShellError = $derived(parseExecShellCommandError(section.toolResult));
	const execShellExitStatus: ExecShellExitStatus | undefined = $derived(
		parseExecShellCommandExitStatus(section.toolResult)
	);

	const parsedLines: ToolResultLine[] = $derived(
		section.toolResult ? parseToolResultWithImages(section.toolResult, attachments) : []
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

	const exitBadgeClass = $derived(
		execShellExitStatus?.timedOut
			? 'exit-badge warning'
			: execShellExitStatus?.code === 0
				? 'exit-badge success'
				: 'exit-badge failure'
	);

	const useFullHeightCodeBlocks = $derived(
		Boolean(config()[SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS])
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
			childList: true,
			subtree: true,
			characterData: true
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

{#snippet execShellTitle()}
	{#if highlightedCommandHtml}
		<span class="font-mono">{@html highlightedCommandHtml}</span>
	{:else}
		<span class="font-mono">{execShellMeta?.command}</span>
	{/if}
{/snippet}

<ToolCallBlock
	{section}
	{open}
	{isStreaming}
	meta={execShellMeta ? { errorMessage: execShellError } : null}
	wrapper={CollapsibleTerminalBlock}
	extraLiveStreaming={isLive}
	spinIconWhenActive={true}
	{onToggle}
>
	{#snippet titleSnippet()}
		{@render execShellTitle()}
	{/snippet}

	{#snippet children(_meta, ctx)}
		{#if ctx.isPending}
			<div class="flex items-start gap-2 text-xs text-muted-foreground/70">
				<Loader2 class="h-3 w-3 animate-spin" />
				Running...
			</div>
		{:else if execShellError}
			<div class="flex items-start gap-2 text-xs text-red-600 italic dark:text-red-400">
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />
				<span>{execShellError}</span>
			</div>
		{:else if section.toolResult}
			<div
				bind:this={scrollEl}
				class="terminal-output"
				class:is-clamped={!useFullHeightCodeBlocks}
				onscroll={handleScrollEvent}
			>
				{#each outputLines as line, i (i)}
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

				{#if isExitCodeFinalLine && execShellExitStatus}
					<div class={exitBadgeClass}>
						{#if execShellExitStatus.timedOut}
							<AlertTriangle class="h-3 w-3" />
							<span>timed out</span>
							<span class="exit-sep">&middot;</span>
							<span>exit {execShellExitStatus.code}</span>
						{:else if execShellExitStatus.code === 0}
							<Check class="h-3 w-3" />
							<span>exit 0</span>
						{:else}
							<XCircle class="h-3 w-3" />
							<span>exit {execShellExitStatus.code}</span>
						{/if}
					</div>
				{/if}
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>

<style>
	.terminal-output {
		overscroll-behavior: contain;
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

	.exit-badge.success {
		background: color-mix(in oklch, var(--color-green-500, #22c55e) 14%, transparent);
		color: var(--color-green-700, #15803d);
	}

	:global(.dark) .exit-badge.success {
		background: color-mix(in oklch, var(--color-green-400, #4ade80) 18%, transparent);
		color: var(--color-green-300, #86efac);
	}

	.exit-badge.failure {
		background: color-mix(in oklch, var(--color-red-500, #ef4444) 14%, transparent);
		color: var(--color-red-700, #b91c1c);
	}

	:global(.dark) .exit-badge.failure {
		background: color-mix(in oklch, var(--color-red-400, #f87171) 18%, transparent);
		color: var(--color-red-300, #fca5a5);
	}

	.exit-badge.warning {
		background: color-mix(in oklch, var(--color-amber-500, #f59e0b) 14%, transparent);
		color: var(--color-amber-700, #b45309);
	}

	:global(.dark) .exit-badge.warning {
		background: color-mix(in oklch, var(--color-amber-400, #fbbf24) 18%, transparent);
		color: var(--color-amber-300, #fcd34d);
	}

	.exit-sep {
		opacity: 0.45;
	}
</style>
