<script lang="ts">
	// One row for a run of consecutive tool calls: a summary of what they did
	// ("Ran 3 commands, read 2 files") or, while one runs, what is happening
	// now ("Reading package.json"). Images the calls produced show as small
	// thumbnails under it. Expanding it lists each call as its own row.

	import { describeToolActivity, summarizeToolCalls } from './parsers/tool-call-phrasing';
	import ToolCallElapsed from './ToolCallElapsed.svelte';
	import { CollapsibleContentBlock } from '#lib/components/app/index.ts';
	import { ScrollCarousel } from '#lib/components/app/index.ts';
	import { ScrollCarouselVariant } from '#lib/enums/index.ts';
	import { mediaGalleryStore, settingsStore } from '#lib/stores/index.ts';
	import type { AgenticSection } from '#lib/types/index.ts';
	import { imageExtras, toolCallShownArg } from '#lib/utils/index.ts';
	import type { Snippet } from 'svelte';
	import { fade } from 'svelte/transition';

	interface Props {
		sections: AgenticSection[];
		open: boolean;
		isStreaming: boolean;
		executingToolCallId?: string | null;
		/** Start of the group's loop (ms since epoch), or null for no timer. */
		elapsedStartedAt?: number | null;
		/** End of the group's loop (ms since epoch), or null while it runs. */
		elapsedEndedAt?: number | null;
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
		elapsedEndedAt = null,
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

	const images = $derived(sections.flatMap((section) => imageExtras(section.toolResultExtras)));
</script>

<div class="tool-call-group min-w-0">
	<CollapsibleContentBlock {onToggle} {open} pinnedChevron shimmerTitle={Boolean(activity)}>
		{#snippet titleSnippet()}
			<span class="flex min-w-0 items-baseline gap-x-2">
				<!-- collapsed, the summary steps back behind the answer; a live title keeps its shimmer -->
				<span
					class={['min-w-0 truncate', !open && !activity && 'font-normal text-muted-foreground']}
					title={summary}>{summary}</span
				>
			</span>
		{/snippet}

		{#snippet trailing()}
			{#if elapsedStartedAt !== null}
				<ToolCallElapsed endedAt={elapsedEndedAt} startedAt={elapsedStartedAt} />
			{/if}
		{/snippet}

		<div class="tool-call-list grid min-w-0">
			{@render children()}
		</div>
	</CollapsibleContentBlock>

	<!-- every image the calls produced, at their own proportions, under the summary
     while the group is closed;
     opening it shows each call with its own images, so the strip fades out -->
	{#if images.length > 0 && !open}
		<div transition:fade={{ duration: 150 }} class="tool-call-media">
			<ScrollCarousel gapSize="2" variant={ScrollCarouselVariant.CENTER}>
				{#each images as image, i (i)}
					<button
						aria-label={image.name ? `Preview ${image.name}` : 'Preview image'}
						class="h-16 shrink-0 overflow-hidden rounded-md border border-border/60 transition-opacity hover:opacity-80"
						onclick={() => mediaGalleryStore.show(image)}
						type="button"
					>
						<img
							alt=""
							class="h-full w-auto max-w-60 object-cover"
							loading="lazy"
							src={image.base64Url}
						/>
					</button>
				{/each}
			</ScrollCarousel>
		</div>
	{/if}
</div>

<style>
	/* under the title text, past the indent the rows use */
	.tool-call-media {
		padding-top: 0.25rem;
	}

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
