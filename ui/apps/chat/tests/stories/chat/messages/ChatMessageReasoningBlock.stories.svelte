<script lang="ts" module>
	import {
		reasoningInterruptedSection,
		reasoningPendingSection,
		reasoningSection,
		seedChatStory
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageReasoningBlock from '#lib/components/app/chat/ChatMessages/ChatMessageReasoningBlock.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	const { Story } = defineMeta({
		component: ChatMessageReasoningBlock,
		parameters: {
			layout: 'padded'
		},
		title: 'Llama Chat/Chat/ChatMessages/ChatMessageReasoningBlock'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="States">
	<StateGallery columns={1}>
		<StateGalleryItem label="Collapsed">
			<ChatMessageReasoningBlock isStreaming={false} open={false} section={reasoningSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Expanded" note="markdown">
			<ChatMessageReasoningBlock isStreaming={false} open section={reasoningSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Streaming">
			<ChatMessageReasoningBlock isStreaming open section={reasoningPendingSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Cancelled">
			<ChatMessageReasoningBlock
				isStreaming={false}
				open={false}
				section={reasoningInterruptedSection}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Error">
			<ChatMessageReasoningBlock
				hasReasoningError
				isStreaming={false}
				open={false}
				section={reasoningInterruptedSection}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>

<!-- Separate: turns off the global renderThinkingAsMarkdown setting. -->
<Story
	args={{ isStreaming: false, open: true, section: reasoningSection }}
	beforeEach={() => seedChatStory({ settings: { renderThinkingAsMarkdown: false } })}
	name="ExpandedPlainText"
/>
