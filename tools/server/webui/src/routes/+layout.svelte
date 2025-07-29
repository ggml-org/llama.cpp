<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import ChatSidebar from '$lib/components/chat/ChatSidebar/ChatSidebar.svelte';
	import { activeMessages, isLoading } from '$lib/stores/chat.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { page } from '$app/state';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';

	let { children } = $props();

	const isHomeRoute = $derived(page.route.id === '/');
	const isChatRoute = $derived(page.route.id === '/chat/[id]');
	const isNewChatMode = $derived(page.url.searchParams.get('new_chat') === 'true');
	const showSidebarByDefault = $derived(activeMessages().length > 0 || isLoading());

	let sidebarOpen = $state(false);

	$effect(() => {
		if (isHomeRoute && !isNewChatMode) {
			// Auto-collapse sidebar when navigating to home route (but not in new chat mode)
			sidebarOpen = false;
		} else if (isHomeRoute && isNewChatMode) {
			// Keep sidebar open in new chat mode
			sidebarOpen = true;
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

<Toaster richColors />

<Sidebar.Provider bind:open={sidebarOpen}>
	<div class="flex h-screen w-full">
		<Sidebar.Root class="h-full">
			<ChatSidebar />
		</Sidebar.Root>

		<Sidebar.Trigger class="z-50 h-8 w-8" style="translate: 1rem 1rem" />

		<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
			{@render children?.()}
		</Sidebar.Inset>
	</div>
</Sidebar.Provider>
