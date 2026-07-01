<script lang="ts">
	import { ScanText } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';
	import {
		ChatMessageActionIcons,
		ChatMessageEditForm,
		ChatMessageMcpPromptContent
	} from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { ChatMessageEditFormVariant, McpPromptVariant, MessageRole } from '$lib/enums';
	import { parseMcpPromptId } from '$lib/utils';
	import type {
		DatabaseMessage,
		DatabaseMessageExtraSkill,
		DatabaseMessageExtraMcpPrompt
	} from '$lib/types';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		skillExtra: DatabaseMessageExtraSkill | null;
		messageRole: MessageRole;
		skillName?: string;
		skillIsStale?: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		showDeleteDialog: boolean;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		onCopy: () => void;
		onEdit: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		onSkillUpdate?: () => void;
	}

	let {
		class: className = '',
		message,
		skillExtra,
		messageRole,
		skillName,
		skillIsStale = false,
		siblingInfo = null,
		showDeleteDialog,
		deletionInfo,
		onCopy,
		onEdit,
		onDelete,
		onConfirmDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange,
		onSkillUpdate
	}: Props = $props();

	const editCtx = getMessageEditContext();

	// Detect whether this skill attachment originates from an MCP prompt.
	// MCP prompts encode server/prompt identity in a synthetic skillId of
	// the form `mcp:<serverName>:<promptName>`.
	let isMcpPrompt = $derived(skillExtra !== null && parseMcpPromptId(skillExtra.skillId) !== null);

	let mcpPrompt: DatabaseMessageExtraMcpPrompt | null = $derived.by(() => {
		if (!isMcpPrompt || !skillExtra) return null;
		const parsed = parseMcpPromptId(skillExtra.skillId);
		if (!parsed) return null;

		return {
			type: 'MCP_PROMPT',
			name: parsed.promptName,
			serverName: parsed.serverName,
			promptName: parsed.promptName,
			content: message.content
		} as DatabaseMessageExtraMcpPrompt;
	});

	let ariaLabel = $derived(
		isMcpPrompt
			? 'MCP Prompt message with actions'
			: messageRole === MessageRole.SYSTEM
				? 'Skill system message with actions'
				: 'Skill message with actions'
	);
</script>

<div
	aria-label={ariaLabel}
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
	data-message-role={messageRole}
>
	{#if editCtx.isEditing}
		<div class="flex w-full flex-col items-end gap-2">
			<ChatMessageEditForm variant={ChatMessageEditFormVariant.SYSTEM} />
		</div>
	{:else}
		{#if isMcpPrompt && mcpPrompt}
			<!-- MCP Prompt rendering -->
			<ChatMessageMcpPromptContent
				prompt={mcpPrompt}
				variant={McpPromptVariant.MESSAGE}
				class="w-full max-w-[80%]"
			/>
		{:else if skillExtra && skillExtra.skillId}
			<!-- Regular skill rendering -->
			<!-- <div class="max-w-[80%]"> -->
			<Card
				class="overflow-y-auto rounded-[1.125rem] !border-2 !border-dashed !border-border/50 bg-muted px-3.75 py-1.5"
			>
				<div class="flex items-center gap-2">
					<ScanText class="h-3.5 w-3.5 text-muted-foreground" />
					<span class="text-sm font-medium">{skillName ?? skillExtra.title}</span>

					{#if skillIsStale && onSkillUpdate}
						<button
							type="button"
							class="ml-auto text-xs font-medium text-amber-600 hover:underline hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300"
							onclick={onSkillUpdate}
							title="This skill has been modified. Click to save the changes back to the library."
						>
							Modified
						</button>
					{/if}
				</div>
			</Card>
			<!-- </div> -->
		{/if}

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActionIcons
					actionsPosition="right"
					{deletionInfo}
					justify="end"
					{onConfirmDelete}
					{onCopy}
					{onDelete}
					{onEdit}
					{onNavigateToSibling}
					{onShowDeleteDialogChange}
					{siblingInfo}
					{showDeleteDialog}
					role={MessageRole.USER}
				/>
			</div>
		{/if}
	{/if}
</div>
