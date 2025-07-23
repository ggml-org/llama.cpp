<script lang="ts">
	import ChatMessages from './ChatMessages.svelte';
	import ChatForm from './ChatForm.svelte';
	import { chatMessages, isLoading, sendMessage, stopGeneration } from '$lib/stores/chat.svelte';
</script>

<ChatMessages class="mx-auto w-full max-w-[50rem]" messages={chatMessages} {isLoading} />

<ChatForm
	class="mx-auto w-full max-w-[50rem]"
	{isLoading}
	onsend={(message) => sendMessage(message)}
	onstop={() => stopGeneration()}
/>
