<script lang="ts">
	import ChatFormPickerCommand from './ChatFormPickerCommand.svelte';
	import ChatFormPickerMcpPrompts from './ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
	import ChatFormPickerMention from './ChatFormPickerMention.svelte';
	import type {
		ChatFormCommand,
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo
	} from '$lib/types';

	interface Props {
		/** What every picker's popover floats by. */
		anchor?: HTMLElement | null;
		isCommandPickerOpen?: boolean;
		commandQuery?: string;
		commands?: ChatFormCommand[];
		isPromptPickerOpen?: boolean;
		promptSearchQuery?: string;
		isMentionPickerOpen?: boolean;
		mentionQuery?: string;
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
		anchor,
		commandQuery,
		commands = [],
		isCommandPickerOpen,
		isMentionPickerOpen,
		isPromptPickerOpen,
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

	let commandPickerRef: ChatFormPickerCommand | undefined = $state(undefined);
	let promptPickerRef: ChatFormPickerMcpPrompts | undefined = $state(undefined);
	let mentionPickerRef: ChatFormPickerMention | undefined = $state(undefined);

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

<ChatFormPickerCommand
	bind:this={commandPickerRef}
	{commands}
	customAnchor={anchor}
	isOpen={isCommandPickerOpen ?? false}
	onClose={onCommandPickerClose ?? (() => {})}
	onSelect={onCommandSelect ?? (() => {})}
	query={commandQuery ?? ''}
/>

<ChatFormPickerMcpPrompts
	bind:this={promptPickerRef}
	customAnchor={anchor}
	isOpen={isPromptPickerOpen}
	onClose={onPromptPickerClose}
	{onPromptLoadComplete}
	{onPromptLoadError}
	{onPromptLoadStart}
	searchQuery={promptSearchQuery}
/>

<ChatFormPickerMention
	bind:this={mentionPickerRef}
	customAnchor={anchor}
	isOpen={isMentionPickerOpen ?? false}
	onClose={onMentionPickerClose ?? (() => {})}
	onOpened={onMentionOpened}
	onSelect={onMentionSelect ?? (() => {})}
	query={mentionQuery ?? ''}
	scopePath={scopePath ?? null}
/>
