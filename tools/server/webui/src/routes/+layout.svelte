<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import ChatSidebar from '$lib/components/chat/ChatSidebar/ChatSidebar.svelte';
	import { chatMessages, isLoading } from '$lib/stores/chat.svelte';
	import { slide } from 'svelte/transition';

	let { children } = $props();

	// Show sidebar only when there are messages or loading
	const hasMessages = $derived(chatMessages.length > 0 || isLoading);
</script>

<ModeWatcher />

<div class="bg-background flex h-screen">
	{#if hasMessages}
		<!-- Sidebar -->
		<aside class="w-72 flex-shrink-0" in:slide={{ duration: 400, axis: 'x' }}>
			<ChatSidebar />
		</aside>
	{/if}

	<!-- Main Content -->
	<main class="flex flex-1 flex-col overflow-hidden">
		{@render children?.()}
	</main>
</div>
