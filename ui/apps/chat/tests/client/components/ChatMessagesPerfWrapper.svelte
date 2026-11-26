<script lang="ts">
	// Mounts the real ChatMessages list against the real conversations store, so
	// the harness exercises `displayMessages` (which rebuilds every message's
	// toolMessages array) rather than a single message subtree.
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatMessages from '#lib/components/app/chat/ChatMessages/ChatMessages.svelte';
	import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
</script>

<Tooltip.Provider>
	<ChatMessages messages={conversationsStore.activeMessages} />
</Tooltip.Provider>
