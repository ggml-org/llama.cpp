<script lang="ts">
	import { BuiltInTool } from '$lib/enums';
	import {
		extractSearchQuery,
		extractSearchResults,
		isWebSearchToolName,
		type AgenticSection
	} from '$lib/utils';
	import type { DatabaseMessageExtra } from '$lib/types';
	import ChatMessageToolCallBlockDefault from './ChatMessageToolCallBlockDefault.svelte';
	import ChatMessageToolCallBlockEditFile from './ChatMessageToolCallBlockEditFile.svelte';
	import ChatMessageToolCallBlockExecShellCommand from './ChatMessageToolCallBlockExecShellCommand.svelte';
	import ChatMessageToolCallBlockFileGlobSearch from './ChatMessageToolCallBlockFileGlobSearch.svelte';
	import ChatMessageToolCallBlockGetDatetime from './ChatMessageToolCallBlockGetDatetime.svelte';
	import ChatMessageToolCallBlockGrepSearch from './ChatMessageToolCallBlockGrepSearch.svelte';
	import ChatMessageToolCallBlockReadFile from './ChatMessageToolCallBlockReadFile.svelte';
	import ChatMessageToolCallBlockRunJavascript from './ChatMessageToolCallBlockRunJavascript.svelte';
	import ChatMessageToolCallBlockSearchResults from './ChatMessageToolCallBlockSearchResults.svelte';
	import ChatMessageToolCallBlockWriteFile from './ChatMessageToolCallBlockWriteFile.svelte';

	interface Props {
		section: AgenticSection;
		attachments?: DatabaseMessageExtra[];
		open: boolean;
		isStreaming: boolean;
		/** True while the agentic loop is streaming output for THIS specific
		 *  tool call (matched by toolCallId). Lets the underlying renderer -
		 *  currently only exec_shell_command - switch into live-update mode
		 *  with auto-scroll + max-height so chunked output stays visible. */
		isExecuting?: boolean;
		onToggle?: () => void;
	}

	let { section, attachments, open, isStreaming, isExecuting, onToggle }: Props = $props();

	// Search-result runs render via the dedicated hover-card block even
	// outside the BuiltInTool namespace. The block already handles the
	// pending state (formatted heading from query + "Searching..." spinner),
	// so we route off `toolArgs.query` *only* when the tool name marks it as
	// a web-search call - otherwise non-search tools that happen to accept
	// a `query` argument (e.g. GitHub's `search_pull_requests`) get
	// mis-labelled. After results arrive the parsed-result check takes
	// over so still-unrecognized servers render correctly.
	const searchResults = $derived(extractSearchResults(section.toolResult));
	const searchQuery = $derived(extractSearchQuery(section.toolArgs));
	const isSearchCall = $derived(
		searchResults.length > 0 || (searchQuery.length > 0 && isWebSearchToolName(section.toolName))
	);
</script>

{#if isSearchCall}
	<ChatMessageToolCallBlockSearchResults {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.GET_DATETIME}
	<ChatMessageToolCallBlockGetDatetime {section} {isStreaming} />
{:else if section.toolName === BuiltInTool.READ_FILE}
	<ChatMessageToolCallBlockReadFile {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.EDIT_FILE}
	<ChatMessageToolCallBlockEditFile {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.WRITE_FILE}
	<ChatMessageToolCallBlockWriteFile {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.EXEC_SHELL_COMMAND}
	<ChatMessageToolCallBlockExecShellCommand
		{section}
		{open}
		{isStreaming}
		{isExecuting}
		{attachments}
		{onToggle}
	/>
{:else if section.toolName === BuiltInTool.FILE_GLOB_SEARCH}
	<ChatMessageToolCallBlockFileGlobSearch {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.GREP_SEARCH}
	<ChatMessageToolCallBlockGrepSearch {section} {open} {isStreaming} {onToggle} />
{:else if section.toolName === BuiltInTool.RUN_JAVASCRIPT}
	<ChatMessageToolCallBlockRunJavascript {section} {open} {isStreaming} {onToggle} />
{:else}
	<ChatMessageToolCallBlockDefault {section} {open} {isStreaming} {attachments} {onToggle} />
{/if}
