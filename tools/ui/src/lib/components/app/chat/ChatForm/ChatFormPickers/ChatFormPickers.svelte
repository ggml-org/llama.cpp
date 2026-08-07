<script lang="ts">
	import ChatFormMentionPicker from './ChatFormMentionPicker.svelte';
	import ChatFormPickerMcpPrompts from './ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
	import type { FileMentionEntry, GetPromptResult, MCPPromptInfo } from '$lib/types';

	interface Props {
		isPromptPickerOpen?: boolean;
		promptSearchQuery?: string;
		isMentionPickerOpen?: boolean;
		mentionQuery?: string;
		mentionAnchor?: HTMLElement | null;
		scopePath?: string | null;
		onPromptPickerClose?: () => void;
		onMentionPickerClose?: () => void;
		onMentionOpened?: () => void;
		onMentionSelect?: (entry: FileMentionEntry) => void;
		onPromptLoadStart?: (
			placeholderId: string,
			promptInfo: MCPPromptInfo,
			args?: Record<string, string>
		) => void;
		onPromptLoadComplete?: (placeholderId: string, result: GetPromptResult) => void;
		onPromptLoadError?: (placeholderId: string, error: string) => void;
	}

	let {
		isPromptPickerOpen,
		promptSearchQuery,
		isMentionPickerOpen,
		mentionQuery,
		mentionAnchor,
		scopePath,
		onPromptPickerClose,
		onMentionPickerClose,
		onMentionOpened,
		onMentionSelect,
		onPromptLoadStart,
		onPromptLoadComplete,
		onPromptLoadError
	}: Props = $props();

	let promptPickerRef: ChatFormPickerMcpPrompts | undefined = $state(undefined);
	let mentionPickerRef: ChatFormMentionPicker | undefined = $state(undefined);

	/**
	 * Delegates keyboard events to the active picker child.
	 * Returns true if the event was handled.
	 */
	export function handleKeydown(event: KeyboardEvent): boolean {
		if (isPromptPickerOpen && promptPickerRef?.handleKeydown(event)) {
			return true;
		}

		if (isMentionPickerOpen && mentionPickerRef?.handleKeydown(event)) {
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

<ChatFormMentionPicker
	bind:this={mentionPickerRef}
	isOpen={isMentionPickerOpen ?? false}
	query={mentionQuery ?? ''}
	customAnchor={mentionAnchor}
	scopePath={scopePath ?? null}
	onClose={onMentionPickerClose ?? (() => {})}
	onOpened={onMentionOpened}
	onSelect={onMentionSelect ?? (() => {})}
/>
