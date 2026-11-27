<script lang="ts">
	import { perfState } from './agentic-perf-state.svelte';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatMessageAgentTrace from '#lib/components/app/chat/ChatMessages/ChatMessageAgentTrace.svelte';
</script>

<Tooltip.Provider>
	{#if perfState.message}
		<ChatMessageAgentTrace
			isLastAssistantMessage
			isStreaming={perfState.isStreaming}
			message={perfState.message}
			toolMessages={perfState.toolMessages}
		/>
	{/if}
</Tooltip.Provider>
