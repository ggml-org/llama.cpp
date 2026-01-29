<script lang="ts">
	import { untrack } from 'svelte';
	import { PROCESSING_INFO_TIMEOUT } from '$lib/constants/processing-info';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { activeMessages, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';

	const processingState = useProcessingState();

	let isCurrentConversationLoading = $derived(isLoading());
	let isStreaming = $derived(isChatStreaming());
	let hasProcessingData = $derived(processingState.processingState !== null);
	let processingDetails = $derived(processingState.getTechnicalDetails());

	let showProcessingInfo = $derived(
		isCurrentConversationLoading || isStreaming || config().keepStatsVisible || hasProcessingData
	);

	$effect(() => {
		const conversation = activeConversation();

		untrack(() => chatStore.setActiveProcessingConversation(conversation?.id ?? null));
	});

	$effect(() => {
		const keepStatsVisible = config().keepStatsVisible;
		const shouldMonitor = keepStatsVisible || isCurrentConversationLoading || isStreaming;

		if (shouldMonitor) {
			processingState.startMonitoring();
		}

		if (!isCurrentConversationLoading && !isStreaming && !keepStatsVisible) {
			const timeout = setTimeout(() => {
				if (!config().keepStatsVisible && !isChatStreaming()) {
					processingState.stopMonitoring();
				}
			}, PROCESSING_INFO_TIMEOUT);

			return () => clearTimeout(timeout);
		}
	});

	$effect(() => {
		const conversation = activeConversation();
		const messages = activeMessages() as DatabaseMessage[];
		const keepStatsVisible = config().keepStatsVisible;

		if (keepStatsVisible && conversation) {
			if (messages.length === 0) {
				untrack(() => chatStore.clearProcessingState(conversation.id));
				return;
			}

			if (!isCurrentConversationLoading && !isStreaming) {
				untrack(() => chatStore.restoreProcessingStateFromMessages(messages, conversation.id));
			}
		}
	});
</script>

<div class="chat-processing-info-container pointer-events-none" class:visible={showProcessingInfo}>
	<div class="chat-processing-info-content">
		{#each processingDetails as detail (detail)}
			<span class="chat-processing-info-detail pointer-events-auto backdrop-blur-sm">{detail}</span>
		{/each}
	</div>
</div>

<style>
	.chat-processing-info-container {
		position: sticky;
		top: 0;
		z-index: 10;
		padding: 0 1rem 0.75rem;
		opacity: 0;
		transform: translateY(50%);
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.chat-processing-info-container.visible {
		opacity: 1;
		transform: translateY(0);
	}

	.chat-processing-info-content {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1rem;
		justify-content: center;
		max-width: 48rem;
		margin: 0 auto;
	}

	.chat-processing-info-detail {
		color: var(--muted-foreground);
		font-size: 0.75rem;
		padding: 0.25rem 0.75rem;
		border-radius: 0.375rem;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace;
		white-space: nowrap;
	}

	@media (max-width: 768px) {
		.chat-processing-info-content {
			gap: 0.5rem;
		}

		.chat-processing-info-detail {
			font-size: 0.7rem;
			padding: 0.2rem 0.5rem;
		}
	}
</style>
