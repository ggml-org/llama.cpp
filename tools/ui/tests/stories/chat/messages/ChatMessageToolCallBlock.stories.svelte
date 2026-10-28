<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import {
		defaultToolMarkdownSection,
		defaultToolPendingSection,
		defaultToolSection,
		defaultToolStreamingSection,
		editFileErrorSection,
		editFileSection,
		execShellFailedSection,
		execShellPendingSection,
		execShellSection,
		execShellTimedOutSection,
		fileGlobSearchSection,
		getDatetimeSection,
		getInfoSection,
		grepSearchSection,
		readFileSection,
		readMediaSection,
		runJavascriptErrorSection,
		runJavascriptSection,
		searchResultsSection,
		seedChatStory,
		writeFileSection
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageToolCallBlock from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
	import { AgenticSectionType } from '$lib/enums';

	const { Story } = defineMeta({
		component: ChatMessageToolCallBlock,
		parameters: {
			layout: 'padded'
		},
		title: 'Chat/ChatScreen/ChatMessages/ChatMessageToolCallBlock'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="States">
	<StateGallery columns={1} title="Generic block">
		<StateGalleryItem label="Default">
			<ChatMessageToolCallBlock isStreaming={false} open section={defaultToolSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Collapsed">
			<ChatMessageToolCallBlock isStreaming={false} open={false} section={defaultToolSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Markdown result">
			<ChatMessageToolCallBlock isStreaming={false} open section={defaultToolMarkdownSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Pending">
			<ChatMessageToolCallBlock isStreaming open section={defaultToolPendingSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Streaming args">
			<ChatMessageToolCallBlock isStreaming open section={defaultToolStreamingSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Truncated" note="no args received">
			<ChatMessageToolCallBlock
				isStreaming={false}
				open
				section={{ ...defaultToolStreamingSection, toolArgs: undefined }}
			/>
		</StateGalleryItem>
	</StateGallery>

	<StateGallery columns={1} title="Per tool">
		<StateGalleryItem label="Get datetime">
			<ChatMessageToolCallBlock isStreaming={false} open section={getDatetimeSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Get datetime, pending">
			<ChatMessageToolCallBlock
				isStreaming
				open
				section={{
					...getDatetimeSection,
					toolResult: undefined,
					type: AgenticSectionType.TOOL_CALL_PENDING
				}}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Get info">
			<ChatMessageToolCallBlock isStreaming={false} open section={getInfoSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Read file">
			<ChatMessageToolCallBlock isStreaming={false} open section={readFileSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Edit file">
			<ChatMessageToolCallBlock isStreaming={false} open section={editFileSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Edit file, error">
			<ChatMessageToolCallBlock isStreaming={false} open section={editFileErrorSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Write file">
			<ChatMessageToolCallBlock isStreaming={false} open section={writeFileSection} />
		</StateGalleryItem>

		<StateGalleryItem label="File glob search">
			<ChatMessageToolCallBlock isStreaming={false} open section={fileGlobSearchSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Grep search">
			<ChatMessageToolCallBlock isStreaming={false} open section={grepSearchSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Run JavaScript">
			<ChatMessageToolCallBlock isStreaming={false} open section={runJavascriptSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Run JavaScript, error">
			<ChatMessageToolCallBlock isStreaming={false} open section={runJavascriptErrorSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Read media">
			<ChatMessageToolCallBlock isStreaming={false} open section={readMediaSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Search results">
			<ChatMessageToolCallBlock isStreaming={false} open section={searchResultsSection} />
		</StateGalleryItem>
	</StateGallery>

	<StateGallery columns={1} title="Shell command">
		<StateGalleryItem label="Completed">
			<ChatMessageToolCallBlock isStreaming={false} open section={execShellSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Failed">
			<ChatMessageToolCallBlock isStreaming={false} open section={execShellFailedSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Timed out">
			<ChatMessageToolCallBlock isStreaming={false} open section={execShellTimedOutSection} />
		</StateGalleryItem>

		<StateGalleryItem label="Running">
			<ChatMessageToolCallBlock
				conversationCwd="/home/user/projects/app"
				isExecuting
				isStreaming
				open
				section={execShellPendingSection}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
