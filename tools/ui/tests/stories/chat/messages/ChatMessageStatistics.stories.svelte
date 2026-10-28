<script lang="ts" module>
	import { agenticTimings } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageStatistics from '$lib/components/app/chat/ChatMessages/ChatMessageStatistics/ChatMessageStatistics.svelte';
	import { ChatMessageStatisticsMode, ChatMessageStatsView } from '$lib/enums';

	const { Story } = defineMeta({
		args: {
			predictedMs: 3120,
			predictedTokens: 164,
			promptMs: 412,
			promptTokens: 1830
		},
		component: ChatMessageStatistics,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageStatistics'
	});
</script>

<Story name="Default" />

<Story args={{ initialView: ChatMessageStatsView.READING }} name="ReadingView" />

<Story args={{ agenticTimings, initialView: ChatMessageStatsView.TOOLS }} name="ToolsView" />

<Story args={{ agenticTimings, initialView: ChatMessageStatsView.SUMMARY }} name="SummaryView" />

<Story args={{ agenticTimings, hideSummary: true }} name="AgenticWithoutSummary" />

<Story args={{ mode: ChatMessageStatisticsMode.GENERATION }} name="GenerationOnly" />

<Story args={{ mode: ChatMessageStatisticsMode.READING }} name="ReadingOnly" />

<Story
	args={{
		isLive: true,
		isProcessingPrompt: true,
		predictedMs: undefined,
		predictedTokens: undefined,
		promptMs: 6000,
		promptTokens: 4200
	}}
	name="LivePromptProcessing"
/>

<Story args={{ isLive: true, predictedMs: 5120, predictedTokens: 128 }} name="LiveGeneration" />
