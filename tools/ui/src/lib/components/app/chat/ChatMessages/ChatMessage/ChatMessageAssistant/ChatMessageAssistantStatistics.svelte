<script lang="ts">
	import { ChatMessageStatistics } from '$lib/components/app';
	import { ChatMessageStatisticsMode } from '$lib/enums';
	import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';

	interface Props {
		message: DatabaseMessage;
		isLoading: boolean;
		processingState: UseProcessingStateReturn;
		showMessageStats: boolean;
	}

	let { message, isLoading, processingState, showMessageStats }: Props = $props();
</script>

{#if showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
	{@const agentic = message.timings.agentic}
	<ChatMessageStatistics
		mode={ChatMessageStatisticsMode.GENERATION}
		promptTokens={agentic ? agentic.llm.prompt_n : message.timings.prompt_n}
		promptMs={agentic ? agentic.llm.prompt_ms : message.timings.prompt_ms}
		predictedTokens={agentic ? agentic.llm.predicted_n : message.timings.predicted_n}
		predictedMs={agentic ? agentic.llm.predicted_ms : message.timings.predicted_ms}
		agenticTimings={agentic}
	/>
{:else if isLoading && showMessageStats}
	{@const liveStats = processingState.getLiveProcessingStats()}
	{@const genStats = processingState.getLiveGenerationStats()}

	{#if genStats}
		<ChatMessageStatistics
			mode={ChatMessageStatisticsMode.GENERATION}
			isLive
			promptTokens={liveStats?.tokensProcessed}
			promptMs={liveStats?.timeMs}
			predictedTokens={genStats.tokensGenerated}
			predictedMs={genStats.timeMs}
		/>
	{/if}
{/if}
