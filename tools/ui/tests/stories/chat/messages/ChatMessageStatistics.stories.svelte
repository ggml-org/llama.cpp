<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { agenticTimings } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageStatistics from '$lib/components/app/chat/ChatMessages/ChatMessageStatistics/ChatMessageStatistics.svelte';
	import { ChatMessageStatisticsMode, ChatMessageStatsView } from '$lib/enums';

	const { Story } = defineMeta({
		component: ChatMessageStatistics,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageStatistics'
	});

	const stats = {
		predictedMs: 3120,
		predictedTokens: 164,
		promptMs: 412,
		promptTokens: 1830
	};
</script>

<Story asChild name="States">
	<StateGallery columns={1}>
		<StateGalleryItem label="Default" note="generation view">
			<ChatMessageStatistics {...stats} />
		</StateGalleryItem>

		<StateGalleryItem label="Reading view">
			<ChatMessageStatistics {...stats} initialView={ChatMessageStatsView.READING} />
		</StateGalleryItem>

		<StateGalleryItem label="Tools view" note="agentic">
			<ChatMessageStatistics {...stats} {agenticTimings} initialView={ChatMessageStatsView.TOOLS} />
		</StateGalleryItem>

		<StateGalleryItem label="Summary view" note="agentic">
			<ChatMessageStatistics
				{...stats}
				{agenticTimings}
				initialView={ChatMessageStatsView.SUMMARY}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Agentic without summary">
			<ChatMessageStatistics {...stats} {agenticTimings} hideSummary />
		</StateGalleryItem>

		<StateGalleryItem label="Generation only">
			<ChatMessageStatistics {...stats} mode={ChatMessageStatisticsMode.GENERATION} />
		</StateGalleryItem>

		<StateGalleryItem label="Reading only">
			<ChatMessageStatistics {...stats} mode={ChatMessageStatisticsMode.READING} />
		</StateGalleryItem>

		<StateGalleryItem label="Live, processing prompt">
			<ChatMessageStatistics
				isLive
				isProcessingPrompt
				predictedMs={undefined}
				predictedTokens={undefined}
				promptMs={6000}
				promptTokens={4200}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Live, generating">
			<ChatMessageStatistics {...stats} isLive predictedMs={5120} predictedTokens={128} />
		</StateGalleryItem>
	</StateGallery>
</Story>
