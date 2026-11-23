<script lang="ts">
	// One row for a run of consecutive tool calls: a summary of what they did
	// ("Ran 3 commands, read 2 files") or, while one runs, what is happening
	// now ("Reading package.json"). Images the calls produced show as
	// thumbnails on the right. Expanding it lists each call as its own row.

	import { describeToolActivity, summarizeToolCalls } from './parsers/tool-call-phrasing';
	import ToolCallElapsed from './ToolCallElapsed.svelte';
	import { CollapsibleContentBlock } from '#lib/components/app/index.ts';
	import { TOOL_CALL_GROUP_MAX_THUMBNAILS } from '#lib/constants/index.ts';
	import { AttachmentType } from '#lib/enums/index.ts';
	import { settingsStore } from '#lib/stores/index.ts';
	import type { AgenticSection, DatabaseMessageExtraImageFile } from '#lib/types/index.ts';
	import { toolCallShownArg } from '#lib/utils/index.ts';
	import type { Snippet } from 'svelte';

	interface Props {
		sections: AgenticSection[];
		open: boolean;
		isStreaming: boolean;
		executingToolCallId?: string | null;
		/** Task start (ms since epoch) for the live timer, or null for none. */
		elapsedStartedAt?: number | null;
		/**
		 * What the server is doing while the reply waits on it ( "Processing 40%",
		 * "Loading weights" ), shown as the live title when this is the last row.
		 */
		status?: string | null;
		onToggle?: () => void;
		children: Snippet;
	}

	let {
		children,
		elapsedStartedAt = null,
		executingToolCallId,
		isStreaming,
		onToggle,
		open,
		sections,
		status = null
	}: Props = $props();

	const activity = $derived(
		status ||
			describeToolActivity(sections, isStreaming, executingToolCallId, (toolName) =>
				toolCallShownArg(settingsStore.config, toolName)
			)
	);
	const summary = $derived(activity ?? summarizeToolCalls(sections));

	const images = $derived(
		sections.flatMap((section) =>
			(section.toolResultExtras ?? []).filter(
				(extra): extra is DatabaseMessageExtraImageFile => extra.type === AttachmentType.IMAGE
			)
		)
	);
	const thumbnails = $derived(images.slice(0, TOOL_CALL_GROUP_MAX_THUMBNAILS));
	const hiddenImages = $derived(images.length - thumbnails.length);
</script>

<CollapsibleContentBlock
	class="my-2"
	inlineChevron
	{onToggle}
	{open}
	shimmerTitle={Boolean(activity)}
>
	{#snippet titleSnippet()}
		<span class="flex min-w-0 items-baseline gap-x-2">
			<!-- collapsed, the summary steps back behind the answer; a live title keeps its shimmer -->
			<span
				class={['min-w-0 truncate', !open && !activity && 'font-normal text-muted-foreground']}
				title={summary}>{summary}</span
			>
		</span>
	{/snippet}

	{#snippet titleSuffix()}
		{#if elapsedStartedAt !== null}
			<ToolCallElapsed startedAt={elapsedStartedAt} />
		{/if}
	{/snippet}

	{#snippet trailing()}
		{#if thumbnails.length > 0}
			<span aria-label="{images.length} images" class="flex items-center gap-1.5">
				{#if hiddenImages > 0}
					<span class="text-caption text-muted-foreground">+{hiddenImages}</span>
				{/if}

				{#each thumbnails as image, i (i)}
					<img
						alt=""
						class="size-6 rounded-sm border border-border/60 object-cover"
						loading="lazy"
						src={image.base64Url}
					/>
				{/each}
			</span>
		{/if}
	{/snippet}

	<div class="tool-call-list grid min-w-0">
		{@render children()}
	</div>
</CollapsibleContentBlock>

<style>
	.tool-call-list > :global(*) {
		min-width: 0;
		margin: 0 !important;
	}

	/* the group already draws the rule; a row's body only indents */
	.tool-call-list :global([data-slot='collapsible-body']) {
		border-left: 0;
		padding-left: 0.5rem;
		padding-bottom: 0;
	}
</style>
