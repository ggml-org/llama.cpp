<script lang="ts">
	import ChatMessages from '$lib/components/ChatMessages.svelte';
	import ChatForm from '$lib/components/ChatForm.svelte';
	import { chatMessages, isLoading, sendMessage, stopGeneration } from '$lib/stores/chat.svelte';
</script>

<div class="flex h-screen flex-col">
	<header class="bg-background border-b p-4">
		<h1 class="text-xl font-semibold">Chat with llama.cpp</h1>
	</header>

	<main class="flex-1 overflow-hidden">
		<ChatMessages messages={chatMessages} isLoading={isLoading} />
	</main>

	<footer>
		<ChatForm
			isLoading={isLoading}
			onsend={(message) => sendMessage(message)}
			onstop={() => stopGeneration()}
		/>
	</footer>
</div>
