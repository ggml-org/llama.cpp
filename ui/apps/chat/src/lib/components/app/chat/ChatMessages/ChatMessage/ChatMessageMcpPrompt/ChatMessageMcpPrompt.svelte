<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageEditForm,
		ChatMessageMcpPromptContent
	} from '#lib/components/app/index.ts';
	import { getChatMessageEditContext } from '#lib/contexts/index.ts';
	import { McpPromptVariant } from '#lib/enums/index.ts';
	import type { DatabaseMessageExtraMcpPrompt } from '#lib/types/index.ts';

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
				<ChatMessageActionIcons actionsPosition="right" justify="end" />
			</div>
		{/if}
	{/if}
</div>
