<script lang="ts">
	import ChatFormPickerMcpPrompts from './ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
	import ChatFormPickerMcpResources from './ChatFormPickerMcpResources.svelte';
	import type { GetPromptResult, MCPPromptInfo } from '$lib/types';

	interface Props {
		isPromptPickerOpen?: boolean;
		promptSearchQuery?: string;
		isInlineResourcePickerOpen?: boolean;
		resourceSearchQuery?: string;
		onPromptPickerClose?: () => void;
		onInlineResourcePickerClose?: () => void;
		onInlineResourceSelect?: () => void;
		onPromptLoadStart?: (
			placeholderId: string,
			promptInfo: MCPPromptInfo,
			args?: Record<string, string>
		) => void;
		onPromptLoadComplete?: (placeholderId: string, result: GetPromptResult) => void;
		onPromptLoadError?: (placeholderId: string, error: string) => void;
		onInlineResourceBrowse?: () => void;
	}

	let {
		isPromptPickerOpen,
		promptSearchQuery,
		isInlineResourcePickerOpen,
		resourceSearchQuery,
		onPromptPickerClose,
		onInlineResourcePickerClose,
		onInlineResourceSelect,
		onPromptLoadStart,
		onPromptLoadComplete,
		onPromptLoadError,
		onInlineResourceBrowse
	}: Props = $props();

	let promptPickerRef: ChatFormPickerMcpPrompts | undefined = $state(undefined);
	let resourcePickerRef: ChatFormPickerMcpResources | undefined = $state(undefined);

	/**
	 * Delegates keyboard events to the active picker child.
	 * Returns true if the event was handled.
	 */
	export function handleKeydown(event: KeyboardEvent): boolean {
		if (isPromptPickerOpen && promptPickerRef?.handleKeydown(event)) {
			return true;
		}

		if (isInlineResourcePickerOpen && resourcePickerRef?.handleKeydown(event)) {
			return true;
		}

		return false;
	}
</script>

<ChatFormPickerMcpPrompts
	bind:this={promptPickerRef}
	isOpen={isPromptPickerOpen}
	searchQuery={promptSearchQuery}
	onClose={onPromptPickerClose}
	{onPromptLoadStart}
	{onPromptLoadComplete}
	{onPromptLoadError}
/>

<ChatFormPickerMcpResources
	bind:this={resourcePickerRef}
	isOpen={isInlineResourcePickerOpen}
	searchQuery={resourceSearchQuery}
	onClose={onInlineResourcePickerClose}
	onResourceSelect={onInlineResourceSelect}
	onBrowse={onInlineResourceBrowse}
/>
