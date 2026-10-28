<script lang="ts" module>
	import {
		reasoningInterruptedSection,
		reasoningPendingSection,
		reasoningSection,
		seedChatStory
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageReasoningBlock from '$lib/components/app/chat/ChatMessages/ChatMessageReasoningBlock.svelte';

	const { Story } = defineMeta({
		args: {
			isStreaming: false,
			open: false,
			section: reasoningSection
		},
		beforeEach: () => seedChatStory(),
		component: ChatMessageReasoningBlock,
		parameters: {
			layout: 'padded'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageReasoningBlock'
	});
</script>

<Story name="Default" />

<Story args={{ open: true }} name="Expanded" />

<Story
	args={{ open: true }}
	beforeEach={() => seedChatStory({ settings: { renderThinkingAsMarkdown: false } })}
	name="ExpandedPlainText"
/>

<Story
	args={{ isStreaming: true, open: true, section: reasoningPendingSection }}
	name="Streaming"
/>

<Story args={{ section: reasoningInterruptedSection }} name="Cancelled" />

<Story args={{ hasReasoningError: true, section: reasoningInterruptedSection }} name="Error" />
