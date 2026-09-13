<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageEditForm,
		ChatMessageMcpPromptContent
	} from '$lib/components/app';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { McpPromptVariant, MessageRole } from '$lib/enums';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		mcpPrompt: DatabaseMessageExtraMcpPrompt;
	}

	let { class: className = '', mcpPrompt, message }: Props = $props();

	// Get edit context
	const editCtx = getChatMessageEditContext();
</script>

<div
	aria-label="MCP Prompt message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<ChatMessageMcpPromptContent
			class="w-full max-w-[80%]"
			prompt={mcpPrompt}
			variant={McpPromptVariant.MESSAGE}
		/>

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActionIcons actionsPosition="right" justify="end" role={MessageRole.USER} />
			</div>
		{/if}
	{/if}
</div>
