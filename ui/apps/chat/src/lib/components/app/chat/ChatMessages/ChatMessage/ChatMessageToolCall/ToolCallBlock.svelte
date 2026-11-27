<script generics="TMeta" lang="ts">
	// Generic chrome shell shared by every per-tool block under
	// `ChatMessageToolCall/`. Owns:
	//   - the collapsible wrapper (CollapsibleContentBlock);
	//   - the running state, shown as a shimmering title;
	//   - the status subtitle pill.
	//   - the call's plain-language description, handed to the title.
	// Components supply only their `meta`, a title snippet, and a body
	// snippet - everything around them is this single source of truth.

	import { describeToolCall } from './parsers/tool-call-phrasing';
	import { useToolCallElapsed } from './tool-call-elapsed';
	import ToolCallElapsed from './ToolCallElapsed.svelte';
	import {
		formatToolCallLine,
		isToolCallRunning,
		type ToolCallDescription
	} from '@ggml-org/llama-harness';
	import { CollapsibleContentBlock } from '#lib/components/app/index.ts';
	import { AgenticSectionType } from '#lib/enums/index.ts';
	import { settingsStore } from '#lib/stores/index.ts';
	import type { AgenticSection } from '#lib/types/index.ts';
	import { imageExtras, toolCallShownArg } from '#lib/utils/index.ts';
	import type { Snippet } from 'svelte';
	import { fade } from 'svelte/transition';

	type ToolCallBlockMetaWithError = TMeta & { errorMessage?: string };

	interface ToolCallCtx {
		isStreaming: boolean;
		isPending: boolean;
		isStreamingCall: boolean;
		isCodeStreaming: boolean;
		isInterrupted: boolean;
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
		 * Treat an empty result as interrupted too. For tools that create
		 * their result message before running (`exec_shell_command`), so a
		 * stop mid-run leaves an empty result rather than none.
		 */
		emptyResultIsInterrupted?: boolean;
		/** A fixed title; without it and without a snippet, the description is the title. */
		title?: string;
		/** The title, given the call's description ( "Read a file", "Reading ..." ). */
		titleSnippet?: Snippet<[ToolCallDescription]>;
		onToggle?: () => void;
		children: Snippet<[TMeta | null | undefined, ToolCallCtx]>;
	}

	let {
		children,
		emptyResultIsInterrupted = false,
		extraLiveStreaming = false,
		isStreaming,
		meta,
		onToggle,
		open,
		section,
		title,
		titleSnippet: blockTitle
	}: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.TOOL_CALL_PENDING);
	const isStreamingCall = $derived(section.type === AgenticSectionType.TOOL_CALL_STREAMING);
	const showSpinner = $derived(isPending || (isStreamingCall && isStreaming) || extraLiveStreaming);
	const isCodeStreaming = $derived(isStreaming && (isPending || isStreamingCall));
	// A finished call with no result was stopped before it returned (user
	// abort, or the page went away mid-execution).
	const isInterrupted = $derived(
		section.type === AgenticSectionType.TOOL_CALL &&
			(section.toolResult === undefined || (emptyResultIsInterrupted && !section.toolResult)) &&
			!showSpinner
	);

	// No subtitle while the call is in flight - the spinner already
	// signals activity; only terminal states get a pill.
	function subtitleFor(errorMessage?: string): string | undefined {
		if (showSpinner) return undefined;

		if (errorMessage) return 'failed';

		if (isStreamingCall && !isStreaming) return 'incomplete';

		if (isInterrupted) return 'interrupted';

		return undefined;
	}

	const subtitle = $derived(subtitleFor(meta?.errorMessage));
	const elapsedStartFor = useToolCallElapsed();
	const elapsedStart = $derived(elapsedStartFor(section.toolCallId));

	// images the call produced, as tiny thumbnails beside its title while the
	// row is closed; opened, its body shows them in full
	const images = $derived(imageExtras(section.toolResultExtras));

	// what the call did, or is doing while it runs; described once here for
	// every per-tool title
	const phrase = $derived(
		describeToolCall(
			section.toolName,
			section.toolArgs,
			extraLiveStreaming || isToolCallRunning(section, isStreaming),
			toolCallShownArg(settingsStore.config, section.toolName)
		)
	);
</script>

<CollapsibleContentBlock
	class="my-2"
	{onToggle}
	{open}
	pinnedChevron
	shimmerTitle={showSpinner}
	{subtitle}
>
	{#snippet titleSnippet()}
		{#if blockTitle}
			{@render blockTitle(phrase)}
		{:else}
			{title ?? formatToolCallLine(phrase)}
		{/if}
	{/snippet}

	{#snippet trailing()}
		{#if elapsedStart !== null}
			<ToolCallElapsed startedAt={elapsedStart} />
		{/if}
	{/snippet}

	{#snippet titleSuffix()}
		{#if images.length > 0 && !open}
			<span
				transition:fade={{ duration: 150 }}
				aria-label="{images.length} {images.length === 1 ? 'image' : 'images'}"
				class="ml-1 inline-flex items-center gap-1 align-middle"
				role="img"
			>
				{#each images as image, i (i)}
					<img
						alt=""
						class="h-5 w-auto max-w-10 rounded-sm border border-border/60 object-cover"
						loading="lazy"
						src={image.base64Url}
					/>
				{/each}
			</span>
		{/if}
	{/snippet}

	{@render children(meta, {
		isCodeStreaming,
		isInterrupted,
		isPending,
		isStreaming,
		isStreamingCall
	})}
</CollapsibleContentBlock>
