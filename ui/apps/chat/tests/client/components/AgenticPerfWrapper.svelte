<script lang="ts">
	import { perfState } from './agentic-perf-state.svelte';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatMessageAgenticContent from '#lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte';
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
