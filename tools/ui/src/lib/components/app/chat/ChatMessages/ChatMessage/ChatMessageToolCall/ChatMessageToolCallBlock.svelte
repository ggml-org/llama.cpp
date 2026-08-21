<script lang="ts">
	import ChatMessageToolCallBlockDefault from './ChatMessageToolCallBlockDefault.svelte';
	import ChatMessageToolCallBlockEditFile from './ChatMessageToolCallBlockEditFile.svelte';
	import ChatMessageToolCallBlockExecShellCommand from './ChatMessageToolCallBlockExecShellCommand.svelte';
	import ChatMessageToolCallBlockFileGlobSearch from './ChatMessageToolCallBlockFileGlobSearch.svelte';
	import ChatMessageToolCallBlockGetDatetime from './ChatMessageToolCallBlockGetDatetime.svelte';
	import ChatMessageToolCallBlockGetInfo from './ChatMessageToolCallBlockGetInfo.svelte';
	import ChatMessageToolCallBlockGrepSearch from './ChatMessageToolCallBlockGrepSearch.svelte';
	import ChatMessageToolCallBlockReadFile from './ChatMessageToolCallBlockReadFile.svelte';
	import ChatMessageToolCallBlockReadMedia from './ChatMessageToolCallBlockReadMedia.svelte';
	import ChatMessageToolCallBlockRunJavascript from './ChatMessageToolCallBlockRunJavascript.svelte';
	import ChatMessageToolCallBlockSearchResults from './ChatMessageToolCallBlockSearchResults.svelte';
	import ChatMessageToolCallBlockWriteFile from './ChatMessageToolCallBlockWriteFile.svelte';
	import { BuiltInTool } from '$lib/enums';
	import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';
	import { extractSearchQuery, extractSearchResults, isWebSearchToolName } from '$lib/utils';

	interface Props {
		section: AgenticSection;
		attachments?: DatabaseMessageExtra[];
		open: boolean;
		isStreaming: boolean;
		isExecuting?: boolean;
		onToggle?: () => void;
	}

	let { attachments, isExecuting, isStreaming, onToggle, open, section }: Props = $props();

	const searchResults = $derived(extractSearchResults(section.toolResult));
	const searchQuery = $derived(extractSearchQuery(section.toolArgs));
	const isSearchCall = $derived(
		searchResults.length > 0 || (searchQuery.length > 0 && isWebSearchToolName(section.toolName))
	);
</script>

{#if isSearchCall}
	<ChatMessageToolCallBlockSearchResults {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.BROWSER_GET_DATETIME}
	<ChatMessageToolCallBlockGetDatetime {section} {isStreaming} />
{:else if section.toolName === BuiltInTool.SERVER_GET_INFO}
	<ChatMessageToolCallBlockGetInfo {section} {isStreaming} />
{:else if section.toolName === BuiltInTool.SERVER_READ_FILE}
	<ChatMessageToolCallBlockReadFile {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.BROWSER_READ_MEDIA}
	<ChatMessageToolCallBlockReadMedia {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.SERVER_EDIT_FILE}
	<ChatMessageToolCallBlockEditFile {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.SERVER_WRITE_FILE}
	<ChatMessageToolCallBlockWriteFile {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.SERVER_EXEC_SHELL_COMMAND}
	<ChatMessageToolCallBlockExecShellCommand
		{section}
		{open}
		{isStreaming}
		{isExecuting}
		{attachments}
		{onToggle}
	/>
{:else if section.toolName === BuiltInTool.SERVER_FILE_GLOB_SEARCH}
	<ChatMessageToolCallBlockFileGlobSearch {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.SERVER_GREP_SEARCH}
	<ChatMessageToolCallBlockGrepSearch {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.BROWSER_RUN_JAVASCRIPT}
	<ChatMessageToolCallBlockRunJavascript {section} {open} {isStreaming} {onToggle} />
{:else}
	<ChatMessageToolCallBlockDefault {section} {open} {isStreaming} {attachments} {onToggle} />
{/if}
