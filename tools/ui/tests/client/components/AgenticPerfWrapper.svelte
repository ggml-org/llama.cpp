<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import ChatMessageAgenticContent from '$lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte';
	import { perfState } from './agentic-perf-state.svelte';
</script>

<Tooltip.Provider>
	{#if perfState.message}
		<ChatMessageAgenticContent
			message={perfState.message}
			toolMessages={perfState.toolMessages}
			isStreaming={perfState.isStreaming}
			isLastAssistantMessage
		/>
	{/if}
</Tooltip.Provider>
