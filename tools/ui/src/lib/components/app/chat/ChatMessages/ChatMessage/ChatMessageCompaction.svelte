<script lang="ts">
	import { FoldVertical } from '@lucide/svelte';
	import {
		ChatMessageActionIcons,
		ChatMessageAssistantModel,
		ChatMessageAssistantStatistics,
		ChatMessageEditForm,
		CollapsibleContentBlock,
		MarkdownContent
	} from '$lib/components/app';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, conversationsStore, serverStore, settingsStore } from '$lib/stores';
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
	const isRouter = $derived(serverStore.isRouterMode);
	// The model the server reported for the summary, so a compaction says who
	// wrote it and the selector can re-summarize with another child.
	const displayedModel = $derived(message.model ?? null);

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
</script>

<div class={className}>
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<CollapsibleContentBlock
			bind:open
			class="my-2"
			collapsible={hasContent}
			icon={FoldVertical}
			iconClass="h-3.5 w-3.5"
			onToggle={() => (userChoseOpen = open)}
			shimmerTitle={isSummarizing || !hasContent}
			{title}
			variant="divider"
		>
			<MarkdownContent class="text-muted-foreground" content={message.content} />

			<!-- The summary's own metadata lives with the summary, so the divider
			     stays a one-line headline. -->
			{#if displayedModel}
				<div class="info my-6 grid gap-4 tabular-nums">
					<div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
						<ChatMessageAssistantModel
							{displayedModel}
							isLoading={chatStore.isLoading}
							{isRouter}
							{onRegenerate}
						/>

						<ChatMessageAssistantStatistics
							isLoading={chatStore.isLoading}
							{message}
							{processingState}
							showMessageStats={currentConfig.showMessageStats}
						/>
					</div>
				</div>
			{/if}

			{#if hasContent && !isSummarizing}
				<!-- Copy, edit, fork, regenerate and delete all apply to the summary:
				     regenerate re-runs it in place. Continue stays a chat-turn flow. -->
				<ChatMessageActionIcons actionsPosition="left" justify="start" {onRegenerate} />
			{/if}
		</CollapsibleContentBlock>
	{/if}
</div>
