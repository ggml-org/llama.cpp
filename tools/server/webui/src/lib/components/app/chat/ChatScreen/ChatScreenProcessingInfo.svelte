<script lang="ts">
	import { untrack } from 'svelte';
	import { PROCESSING_INFO_TIMEOUT } from '$lib/constants';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { activeMessages, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { ProcessingInfo } from '$lib/components/app/misc';

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

<ProcessingInfo visible={showProcessingInfo} {processingDetails} />
