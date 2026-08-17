<script lang="ts" generics="TMeta">
	// Shared tool-call chrome; child blocks provide metadata and body content.

	import { Loader2, Wrench } from '@lucide/svelte';
	import { CollapsibleContentBlock } from '$lib/components/app';
	import { ICON_CLASS_DEFAULT, ICON_CLASS_SPIN } from '$lib/constants';
	import { AgenticSectionType } from '$lib/enums';
	import { mcpStore } from '$lib/stores';
	import type { AgenticSection, ToolUiEntry } from '$lib/types';
	import { getToolUi } from '$lib/utils';
	import type { Component, Snippet } from 'svelte';

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
		/** Tool metadata used for the status subtitle. */
		meta: ToolCallBlockMetaWithError | null | undefined;
		/** True while tool output continues after argument streaming. */
		extraLiveStreaming?: boolean;
		/** Renderer icon, overridden by the active spinner when configured. */
		icon?: Component;
		/** Replace the title-row icon with a spinner while active. */
		spinIconWhenActive?: boolean;
		/** Wrapper for the title row and body content. */
		wrapper?: typeof CollapsibleContentBlock;
		title?: string;
		titleSnippet?: Snippet;
		onToggle?: () => void;
		children: Snippet<[TMeta | null | undefined, ToolCallCtx]>;
	}

	let {
		children,
		extraLiveStreaming = false,
		icon,
		isStreaming,
		meta,
		onToggle,
		open,
		section,
		spinIconWhenActive = false,
		title,
		titleSnippet,
		wrapper: Wrapper = CollapsibleContentBlock
	}: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming) || extraLiveStreaming);
	const isCodeStreaming = $derived(isStreaming && (isPending || isStreamingCall));

	const toolUi: ToolUiEntry | null = $derived(getToolUi(section.toolName));
	const toolIcon: Component = $derived(
		spinIconWhenActive && showSpinner ? Loader2 : (icon ?? toolUi?.icon ?? Wrench)
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

	// No subtitle while the call is in flight - the spinner already
	// signals activity; only terminal states get a pill.
	function subtitleFor(errorMessage?: string): string | undefined {
		if (showSpinner) return undefined;

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
		isCodeStreaming,
		isPending,
		isStreaming,
		isStreamingCall
	})}
</Wrapper>
