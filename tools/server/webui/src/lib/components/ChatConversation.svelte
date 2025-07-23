<script lang="ts">
	import ChatMessages from './ChatMessages.svelte';
	import ChatForm from './ChatForm.svelte';
	import { chatMessages, isLoading, sendMessage, stopGeneration } from '$lib/stores/chat.svelte';
</script>

<ChatMessages messages={chatMessages} {isLoading} />

<ChatForm {isLoading} onsend={(message) => sendMessage(message)} onstop={() => stopGeneration()} />
