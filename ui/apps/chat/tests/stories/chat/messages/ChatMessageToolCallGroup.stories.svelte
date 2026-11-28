<script lang="ts" module>
	import {
		seedChatStory,
		TOOL_GROUP_EXECUTING_ID,
		toolGroupImageSections,
		toolGroupMcpSections,
		toolGroupRunningSections,
		toolGroupSections,
		toolGroupShellRunningSections
	} from './fixtures';
	import ToolCallGroupStoryHost from './ToolCallGroupStoryHost.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageToolCallGroup from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallGroup.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	const { Story } = defineMeta({
		component: ChatMessageToolCallGroup,
		parameters: {
			layout: 'padded'
		},
		title: 'Llama Chat/Chat/ChatMessages/ChatMessageToolCallGroup'
	});
</script>

<Story asChild beforeEach={() => seedChatStory()} name="States">
	<StateGallery columns={1} title="Done">
		<StateGalleryItem label="Collapsed" note="a summary of what the calls did">
			<ToolCallGroupStoryHost sections={toolGroupSections} />
		</StateGalleryItem>

		<StateGalleryItem label="Expanded" note="one row per call">
			<ToolCallGroupStoryHost open sections={toolGroupSections} />
		</StateGalleryItem>

		<StateGalleryItem
			label="With images"
			note="a thumbnail per image under the title, a carousel when they overflow"
		>
			<ToolCallGroupStoryHost sections={toolGroupImageSections} />
		</StateGalleryItem>

		<StateGalleryItem label="MCP and on-demand tools" note="tool_search and generic tools">
			<ToolCallGroupStoryHost open sections={toolGroupMcpSections} />
		</StateGalleryItem>
	</StateGallery>

	<StateGallery columns={1} title="Running">
		<StateGalleryItem label="Reading a file" note="what happens now">
			<ToolCallGroupStoryHost isStreaming sections={toolGroupRunningSections} />
		</StateGalleryItem>

		<StateGalleryItem label="Running a command" note="streaming tool, live output">
			<ToolCallGroupStoryHost
				executingToolCallId={TOOL_GROUP_EXECUTING_ID}
				isStreaming
				sections={toolGroupShellRunningSections}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Running a command, expanded">
			<ToolCallGroupStoryHost
				executingToolCallId={TOOL_GROUP_EXECUTING_ID}
				isStreaming
				open
				sections={toolGroupShellRunningSections}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
