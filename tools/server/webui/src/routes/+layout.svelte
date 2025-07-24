<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import ChatSidebar from '$lib/components/chat/ChatSidebar/ChatSidebar.svelte';
	import { activeChatMessages, isLoading } from '$lib/stores/chat.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { page } from '$app/state';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';

	let { children } = $props();

	const isHomeRoute = $derived(page.route.id === '/');
	const isChatRoute = $derived(page.route.id === '/chat/[id]');
	const showSidebarByDefault = $derived(activeChatMessages().length > 0 || isLoading());

	let sidebarOpen = $state(false);

	$effect(() => {
		if (isHomeRoute) {
			// Auto-collapse sidebar when navigating to home route
			sidebarOpen = false;
		} else if (isChatRoute) {
			// On chat routes, show sidebar by default
			sidebarOpen = true;
		} else {
			// Other routes follow default behavior
			sidebarOpen = showSidebarByDefault;
		}
	});

	// Initialize server properties on app load
	$effect(() => {
		serverStore.fetchServerProps();
	});
</script>

<ModeWatcher />

<Sidebar.Provider bind:open={sidebarOpen}>
	<div class="flex h-screen w-full">
		<Sidebar.Root class="h-full">
			<ChatSidebar />
		</Sidebar.Root>

		{#if !isChatRoute}
			<Sidebar.Trigger class="h-8 w-8" style="translate: 0.5rem 0.5rem" />
		{/if}

		<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
			{@render children?.()}
		</Sidebar.Inset>
	</div>
</Sidebar.Provider>
