<script lang="ts" module>
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
		args: {
			isStreaming: false,
			open: true,
			section: defaultToolSection
		},
		beforeEach: () => seedChatStory(),
		component: ChatMessageToolCallBlock,
		parameters: {
			layout: 'padded'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageToolCallBlock'
	});
</script>

<Story name="Default" />

<Story args={{ open: false }} name="Collapsed" />

<Story args={{ section: defaultToolMarkdownSection }} name="MarkdownResult" />

<Story args={{ isStreaming: true, section: defaultToolPendingSection }} name="Pending" />

<Story args={{ isStreaming: true, section: defaultToolStreamingSection }} name="StreamingArgs" />

<Story
	args={{ section: { ...defaultToolStreamingSection, toolArgs: undefined } }}
	name="Truncated"
/>

<Story args={{ section: getDatetimeSection }} name="GetDatetime" />

<Story
	args={{
		isStreaming: true,
		section: {
			...getDatetimeSection,
			toolResult: undefined,
			type: AgenticSectionType.TOOL_CALL_PENDING
		}
	}}
	name="GetDatetimePending"
/>

<Story args={{ section: getInfoSection }} name="GetInfo" />

<Story args={{ section: readFileSection }} name="ReadFile" />

<Story args={{ section: editFileSection }} name="EditFile" />

<Story args={{ section: editFileErrorSection }} name="EditFileError" />

<Story args={{ section: writeFileSection }} name="WriteFile" />

<Story args={{ section: execShellSection }} name="ExecShellCommand" />

<Story args={{ section: execShellFailedSection }} name="ExecShellCommandFailed" />

<Story args={{ section: execShellTimedOutSection }} name="ExecShellCommandTimedOut" />

<Story
	args={{
		conversationCwd: '/home/user/projects/app',
		isExecuting: true,
		isStreaming: true,
		section: execShellPendingSection
	}}
	name="ExecShellCommandRunning"
/>

<Story args={{ section: fileGlobSearchSection }} name="FileGlobSearch" />

<Story args={{ section: grepSearchSection }} name="GrepSearch" />

<Story args={{ section: runJavascriptSection }} name="RunJavascript" />

<Story args={{ section: runJavascriptErrorSection }} name="RunJavascriptError" />

<Story args={{ section: readMediaSection }} name="ReadMedia" />

<Story args={{ section: searchResultsSection }} name="SearchResults" />
