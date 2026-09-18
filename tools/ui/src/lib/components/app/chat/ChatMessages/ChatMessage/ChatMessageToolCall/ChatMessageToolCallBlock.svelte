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
	import {
		extractSearchQuery,
		extractSearchResults,
		isWebSearchToolName,
		looksLikeSearchResult
	} from '$lib/utils';

	interface Props {
		section: AgenticSection;
		attachments?: DatabaseMessageExtra[];
		open: boolean;
		isStreaming: boolean;
		isExecuting?: boolean;
		onToggle?: () => void;
	}

	let { attachments, isExecuting, isStreaming, onToggle, open, section }: Props = $props();

	// Runs for every tool block on mount, before the body renders: the cheap
	// content prefilter and the tool-name allow-list come first so blobs from
	// exec/file tools are never line-split or JSON-parsed here
	const isSearchCall = $derived.by(() => {
		if (looksLikeSearchResult(section.toolResult)) {
			return extractSearchResults(section.toolResult).length > 0;
		}

		return isWebSearchToolName(section.toolName) && extractSearchQuery(section.toolArgs).length > 0;
	});
</script>

{#if isSearchCall}
	<ChatMessageToolCallBlockSearchResults {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.BROWSER_GET_DATETIME}
	<ChatMessageToolCallBlockGetDatetime {isStreaming} {section} />
{:else if section.toolName === BuiltInTool.SERVER_GET_INFO}
	<ChatMessageToolCallBlockGetInfo {isStreaming} {section} />
{:else if section.toolName === BuiltInTool.SERVER_READ_FILE}
	<ChatMessageToolCallBlockReadFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.BROWSER_READ_MEDIA}
	<ChatMessageToolCallBlockReadMedia {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_EDIT_FILE}
	<ChatMessageToolCallBlockEditFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_WRITE_FILE}
	<ChatMessageToolCallBlockWriteFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_EXEC_SHELL_COMMAND}
	<ChatMessageToolCallBlockExecShellCommand
		{attachments}
		{isExecuting}
		{isStreaming}
		{onToggle}
		{open}
		{section}
	/>
{:else if section.toolName === BuiltInTool.SERVER_FILE_GLOB_SEARCH}
	<ChatMessageToolCallBlockFileGlobSearch {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_GREP_SEARCH}
	<ChatMessageToolCallBlockGrepSearch {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.BROWSER_RUN_JAVASCRIPT}
	<ChatMessageToolCallBlockRunJavascript {isStreaming} {onToggle} {open} {section} />
{:else}
	<ChatMessageToolCallBlockDefault {attachments} {isStreaming} {onToggle} {open} {section} />
{/if}
