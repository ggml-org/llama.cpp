<script lang="ts">
	import { perfState } from './agentic-perf-state.svelte';
	import ChatMessageAgenticContent from '$lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
</script>

<Tooltip.Provider>
	{#if perfState.message}
		<ChatMessageAgenticContent
			isLastAssistantMessage
			isStreaming={perfState.isStreaming}
			message={perfState.message}
			toolMessages={perfState.toolMessages}
		/>
	{/if}
</Tooltip.Provider>
