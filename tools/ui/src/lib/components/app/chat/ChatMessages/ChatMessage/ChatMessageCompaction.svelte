<script lang="ts">
	import { FoldVertical, Loader2 } from '@lucide/svelte';
	import {
		ChatMessageActionIcons,
		ChatMessageAssistantStatistics,
		ChatMessageEditForm,
		CollapsibleContentBlock,
		MarkdownContent
	} from '$lib/components/app';
	import { COLLAPSIBLE_TRANSITION_MS } from '$lib/constants';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, conversationsStore, settingsStore } from '$lib/stores';
	import type { DatabaseMessage } from '$lib/types';
	import { compactionTokensBefore, formatNumber } from '$lib/utils';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		onRegenerate: (modelOverride?: string) => void;
	}

	let { class: className = '', message, onRegenerate }: Props = $props();

	const COMPACTION_HEADER = 'Conversation compacted';
	const COMPACTION_RUNNING_LABEL = 'Compacting conversation';
	const COMPACTION_GENERATING_LABEL = 'Generating summary';

	// The shared row context, so the summary is editable and branchable through
	// the same actions as a chat turn.
	const editCtx = getChatMessageEditContext();

	const currentConfig = $derived(settingsStore.config);

	const hasContent = $derived(message.content.trim().length > 0);

	// Only the node a running summary writes to reports progress: a finished
	// compaction further up the branch keeps its headline while another one runs.
	const isSummarizing = $derived(chatStore.getCompactingNodeId(message.convId) === message.id);

	const processingState = useProcessingState();

	// The summary request feeds the shared processing state, so the body shows
	// the same prompt progress line a chat turn does while it waits.
	$effect(() => {
		if (isSummarizing) processingState.startMonitoring();
	});

	// The header carries the run: the prompt progress until the summary starts
	// streaming, the phase after that. There is nothing to collapse before the
	// first summary token arrives.
	const runningLabel = $derived.by(() => {
		// The prompt is through: whatever streams in now is the summary itself.
		if (hasContent) return `${COMPACTION_RUNNING_LABEL} — ${COMPACTION_GENERATING_LABEL}`;

		const progress = processingState.getPromptProgressText();

		return progress
			? `${COMPACTION_RUNNING_LABEL} — ${progress}`
			: `${COMPACTION_RUNNING_LABEL}...`;
	});

	// Shrinking the branch is the feature's own headline, so the finished
	// toggle reports the context size it compacted.
	const tokensBefore = $derived(compactionTokensBefore(conversationsStore.activeMessages, message));

	const completedLabel = $derived(
		tokensBefore === null
			? COMPACTION_HEADER
			: `${COMPACTION_HEADER} from ${formatNumber(tokensBefore)}`
	);

	const title = $derived(isSummarizing || !hasContent ? runningLabel : completedLabel);

	let open = $state(false);
	// A toggle by hand outranks the in-progress expansion, the way the agentic
	// sections treat their own overrides.
	let userChoseOpen = $state<boolean | null>(null);

	// The progress lives in the body (as it does for an assistant turn), so the
	// row opens while the summary runs and collapses once it is done.
	$effect(() => {
		if (userChoseOpen !== null) return;

		open = isSummarizing;
	});

	let rootEl = $state<HTMLDivElement>();
	// The scroll waiting for the body to reach its height, so closing the block
	// before that leaves no scroll behind it.
	let settleTimer: ReturnType<typeof setTimeout> | null = null;

	/**
	 * A toggle hands the reader the block's own boundary, just below the tabs: one
	 * scroll, and only once the body has its height - before that the page is too
	 * short to scroll the divider up there and the call would clamp.
	 */
	function handleToggle() {
		userChoseOpen = open;
		cancelSettledScroll();

		if (!rootEl) return;

		settleTimer = setTimeout(() => {
			settleTimer = null;
			rootEl?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		}, COLLAPSIBLE_TRANSITION_MS);
	}

	function cancelSettledScroll() {
		if (settleTimer === null) return;

		clearTimeout(settleTimer);
		settleTimer = null;
	}

	$effect(() => cancelSettledScroll);
</script>

<div bind:this={rootEl} class="scroll-mt-[calc(var(--chat-tabs-height,0px)+1rem)] {className}">
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<CollapsibleContentBlock
			bind:open
			class="my-2"
			collapsible={hasContent}
			flushBody
			icon={isSummarizing ? Loader2 : FoldVertical}
			iconClass={isSummarizing ? 'h-3.5 w-3.5 animate-spin' : 'h-3.5 w-3.5'}
			onToggle={handleToggle}
			shimmerTitle={isSummarizing || !hasContent}
			{title}
			variant="divider"
		>
			<MarkdownContent class="text-muted-foreground" content={message.content} />

			<!-- The summary's own statistics live with the summary, so the divider
			     stays a one-line headline. Only the conversation model writes the
			     summary, so there is no selector to swap it with. -->
			<div class="info my-6 grid gap-4 tabular-nums">
				<ChatMessageAssistantStatistics
					isLoading={chatStore.isLoading}
					{message}
					{processingState}
					showMessageStats={currentConfig.showMessageStats}
				/>
			</div>

			{#if hasContent && !isSummarizing}
				<!-- Copy, edit, fork, regenerate and delete all apply to the summary:
				     regenerate re-runs it in place. Continue stays a chat-turn flow. -->
				<ChatMessageActionIcons actionsPosition="left" justify="start" {onRegenerate} />
			{/if}
		</CollapsibleContentBlock>
	{/if}
</div>
