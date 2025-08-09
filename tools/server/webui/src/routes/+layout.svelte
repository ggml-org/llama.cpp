<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { ChatSidebar, MaximumContextAlertDialog } from '$lib/components/app';
	import { activeMessages, isLoading } from '$lib/stores/chat.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { serverStore } from '$lib/stores/server.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';

	let { children } = $props();

	let isChatRoute = $derived(page.route.id === '/chat/[id]');
	let isHomeRoute = $derived(page.route.id === '/');
	let isNewChatMode = $derived(page.url.searchParams.get('new_chat') === 'true');
	let showSidebarByDefault = $derived(activeMessages().length > 0 || isLoading());
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

<MaximumContextAlertDialog />

<Sidebar.Provider bind:open={sidebarOpen}>
	<div class="flex h-screen w-full">
		<Sidebar.Root class="h-full">
			<ChatSidebar />
		</Sidebar.Root>

		<Sidebar.Trigger
			class="transition-left absolute h-8 w-8 duration-200 ease-linear {sidebarOpen
				? 'md:left-[var(--sidebar-width)]'
				: 'left-0'}"
			style="translate: 1rem 1rem; z-index: 99999;"
		/>

		<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
			{@render children?.()}
		</Sidebar.Inset>
	</div>
</Sidebar.Provider>
