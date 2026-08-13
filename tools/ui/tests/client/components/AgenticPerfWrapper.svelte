<script lang="ts">
	import { perfState } from './agentic-perf-state.svelte';
	import ChatMessageAgenticContent from '$lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
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
