<script lang="ts">
	// Story-only host: keeps the open state of a tool-call group and of each
	// call in it, as ChatMessageAgentTrace does.
	import ChatMessageToolCallBlock from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
	import ChatMessageToolCallGroup from '#lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallGroup.svelte';
	import type { AgenticSection } from '#lib/types/index.ts';

	interface Props {
		sections: AgenticSection[];
		open?: boolean;
		isStreaming?: boolean;
		executingToolCallId?: string | null;
	}

	let {
		executingToolCallId = null,
		isStreaming = false,
		open: initialOpen = false,
		sections
	}: Props = $props();

	// svelte-ignore state_referenced_locally
	let open = $state(initialOpen);
	let openCalls: Record<number, boolean> = $state({});
</script>

<ChatMessageToolCallGroup
	{executingToolCallId}
	{isStreaming}
	onToggle={() => (open = !open)}
	{open}
	{sections}
>
	{#each sections as section, i (i)}
		{@const executing = section.toolCallId === executingToolCallId}
		<ChatMessageToolCallBlock
			attachments={section.toolResultExtras}
			conversationCwd="/home/user/projects/app"
			isExecuting={executing}
			{isStreaming}
			onToggle={() => (openCalls[i] = !(openCalls[i] ?? executing))}
			open={openCalls[i] ?? executing}
			{section}
		/>
	{/each}
</ChatMessageToolCallGroup>
