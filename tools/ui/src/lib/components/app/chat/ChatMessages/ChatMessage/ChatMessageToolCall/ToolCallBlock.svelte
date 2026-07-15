<script lang="ts" generics="TMeta">
	// Generic chrome shell shared by every per-tool block under
	// `ChatMessageToolCall/`. Owns:
	//   - the collapsible wrapper (defaults to CollapsibleContentBlock;
	//     `exec_shell_command` swaps in CollapsibleTerminalBlock via the
	//     `wrapper` prop);
	//   - the icon, spinner state, and MCP favicon fallback chain;
	//   - the status subtitle pill.
	// Components supply only their `meta`, a title snippet, and a body
	// snippet - everything around them is this single source of truth.

	import { Loader2, Wrench } from '@lucide/svelte';
	import { CollapsibleContentBlock } from '$lib/components/app';
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SPIN } from '$lib/constants/css-classes';
	import { AgenticSectionType } from '$lib/enums';
	import { getBuiltinToolUi } from '$lib/constants/built-in-tools';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { Component, Snippet } from 'svelte';
	import type { AgenticSection, BuiltinToolUiEntry } from '$lib/utils';

	type ToolCallBlockMetaWithError = TMeta & { errorMessage?: string };

	interface ToolCallCtx {
		isStreaming: boolean;
		isPending: boolean;
		isStreamingCall: boolean;
		isCodeStreaming: boolean;
	}

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		/**
		 * The per-tool meta, including any `errorMessage` field that the
		 * shared chrome uses to compute the status pill subtitle.
		 */
		meta: ToolCallBlockMetaWithError | null | undefined;
		/**
		 * True while the tool's process is actively producing output
		 * chunks after its args finished streaming (used by
		 * `exec_shell_command`'s stdout feed).
		 */
		extraLiveStreaming?: boolean;
		/**
		 * Swap the title-row icon for a spinning `Loader2` while the
		 * spinner is showing. Only meaningful for tools where "live"
		 * is interesting (e.g. exec_shell_command showing the in-flight
		 * process). Other tools leave it off and render the spinner
		 * inline within the body.
		 */
		spinIconWhenActive?: boolean;
		/**
		 * Wrapper component that renders the title row and the body
		 * children. Defaults to CollapsibleContentBlock;
		 * `exec_shell_command` uses CollapsibleTerminalBlock for its
		 * terminal-style frame.
		 */
		wrapper?: typeof CollapsibleContentBlock;
		title?: string;
		titleSnippet?: Snippet;
		onToggle?: () => void;
		children: Snippet<[TMeta | null | undefined, ToolCallCtx]>;
	}

	let {
		section,
		open,
		isStreaming,
		meta,
		extraLiveStreaming = false,
		spinIconWhenActive = false,
		wrapper: Wrapper = CollapsibleContentBlock,
		title,
		titleSnippet,
		onToggle,
		children
	}: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming) || extraLiveStreaming);
	const isCodeStreaming = $derived(isStreaming && (isPending || isStreamingCall));

	const toolUi: BuiltinToolUiEntry | null = $derived(getBuiltinToolUi(section.toolName));
	const toolIcon: Component = $derived(
		spinIconWhenActive && showSpinner ? Loader2 : (toolUi?.icon ?? Wrench)
	);
	const toolIconClass = $derived(
		spinIconWhenActive && showSpinner ? ICON_CLASS_SPIN : ICON_CLASS_DEFAULT
	);
	// Drop the MCP favicon while the spinner is on so the title row
	// signals "in flight" without being overwritten by server branding.
	const mcpServerFavicon = $derived(
		showSpinner ? null : mcpStore.getServerFaviconForTool(section.toolName)
	);
	const iconUrl = $derived(
		showSpinner || (toolUi?.icon ?? null) || !mcpServerFavicon ? null : mcpServerFavicon
	);

	function subtitleFor(errorMessage?: string): string | undefined {
		if (extraLiveStreaming) return 'streaming...';
		if (showSpinner) return 'executing...';
		if (errorMessage) return 'failed';
		if (isStreamingCall && !isStreaming) return 'incomplete';
		return undefined;
	}

	const subtitle = $derived(subtitleFor(meta?.errorMessage));
</script>

<Wrapper
	{open}
	class="my-2"
	icon={toolIcon}
	iconClass={toolIconClass}
	{iconUrl}
	{title}
	{titleSnippet}
	{subtitle}
	{onToggle}
>
	{@render children(meta, {
		isStreaming,
		isPending,
		isStreamingCall,
		isCodeStreaming
	})}
</Wrapper>
