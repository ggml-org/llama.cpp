<script lang="ts">
	import ChatFormCommandPicker from './ChatFormCommandPicker.svelte';
	import ChatFormMentionPicker from './ChatFormMentionPicker.svelte';
	import ChatFormPickerMcpPrompts from './ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
	import type {
		ChatFormCommand,
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo
	} from '$lib/types';

	interface Props {
		isCommandPickerOpen?: boolean;
		commandQuery?: string;
		commands?: ChatFormCommand[];
		isPromptPickerOpen?: boolean;
		promptSearchQuery?: string;
		isMentionPickerOpen?: boolean;
		mentionQuery?: string;
		mentionAnchor?: HTMLElement | null;
		scopePath?: string | null;
		onCommandPickerClose?: () => void;
		onCommandSelect?: (command: ChatFormCommand) => void;
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
		commandQuery,
		commands = [],
		isCommandPickerOpen,
		isMentionPickerOpen,
		isPromptPickerOpen,
		mentionAnchor,
		mentionQuery,
		onCommandPickerClose,
		onCommandSelect,
		onMentionOpened,
		onMentionPickerClose,
		onMentionSelect,
		onPromptLoadComplete,
		onPromptLoadError,
		onPromptLoadStart,
		onPromptPickerClose,
		promptSearchQuery,
		scopePath
	}: Props = $props();

	let commandPickerRef: ChatFormCommandPicker | undefined = $state(undefined);
	let promptPickerRef: ChatFormPickerMcpPrompts | undefined = $state(undefined);
	let mentionPickerRef: ChatFormMentionPicker | undefined = $state(undefined);

	/** Delegate keyboard events to the active picker child; true if handled. */
	export function handleKeydown(event: KeyboardEvent): boolean {
		if (isCommandPickerOpen && commandPickerRef?.handleKeydown(event)) {
			return true;
		}

		if (isPromptPickerOpen && promptPickerRef?.handleKeydown(event)) {
			return true;
		}

		if (isMentionPickerOpen && mentionPickerRef?.handleKeydown(event)) {
			return true;
		}

		return false;
	}
</script>

<ChatFormCommandPicker
	bind:this={commandPickerRef}
	isOpen={isCommandPickerOpen ?? false}
	query={commandQuery ?? ''}
	{commands}
	onClose={onCommandPickerClose ?? (() => {})}
	onSelect={onCommandSelect ?? (() => {})}
/>

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
