<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import {
		ChatSidebar,
		ConversationTitleUpdateDialog,
		MaximumContextAlertDialog
	} from '$lib/components/app';
	import {
		activeMessages,
		isLoading,
		setTitleUpdateConfirmationCallback
	} from '$lib/stores/chat.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { serverStore } from '$lib/stores/server.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import { goto } from '$app/navigation';

	let { children } = $props();

	let isChatRoute = $derived(page.route.id === '/chat/[id]');
	let isHomeRoute = $derived(page.route.id === '/');
	let isNewChatMode = $derived(page.url.searchParams.get('new_chat') === 'true');
	let showSidebarByDefault = $derived(activeMessages().length > 0 || isLoading());
	let sidebarOpen = $state(false);
	let innerHeight = $state<number | undefined>();
	let chatSidebar:
		| { activateSearchMode?: () => void; editActiveConversation?: () => void }
		| undefined = $state();

	// Conversation title update dialog state
	let titleUpdateDialogOpen = $state(false);
	let titleUpdateCurrentTitle = $state('');
	let titleUpdateNewTitle = $state('');
	let titleUpdateResolve: ((value: boolean) => void) | null = null;

	// Global keyboard shortcuts
	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (isCtrlOrCmd && event.key === 'k') {
			event.preventDefault();
			if (chatSidebar?.activateSearchMode) {
				chatSidebar.activateSearchMode();
				sidebarOpen = true;
			}
		}

		if (isCtrlOrCmd && event.shiftKey && event.key === 'o') {
			event.preventDefault();
			goto('?new_chat=true#/');
		}

		if (event.shiftKey && isCtrlOrCmd && event.key === 'e') {
			event.preventDefault();

			if (chatSidebar?.editActiveConversation) {
				chatSidebar.editActiveConversation();
			}
		}
	}

	function handleTitleUpdateCancel() {
		titleUpdateDialogOpen = false;
		if (titleUpdateResolve) {
			titleUpdateResolve(false);
			titleUpdateResolve = null;
		}
	}

	function handleTitleUpdateConfirm() {
		titleUpdateDialogOpen = false;
		if (titleUpdateResolve) {
			titleUpdateResolve(true);
			titleUpdateResolve = null;
		}
	}

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

	// Monitor API key changes and redirect to error page if removed or changed when required
	$effect(() => {
		const apiKey = config().apiKey;

		if (
			(page.route.id === '/' || page.route.id === '/chat/[id]') &&
			page.status !== 401 &&
			page.status !== 403
		) {
			const headers: Record<string, string> = {
				'Content-Type': 'application/json'
			};

			if (apiKey && apiKey.trim() !== '') {
				headers.Authorization = `Bearer ${apiKey.trim()}`;
			}

			fetch(`./props`, { headers })
				.then((response) => {
					if (response.status === 401 || response.status === 403) {
						window.location.reload();
					}
				})
				.catch((e) => {
					console.error('Error checking API key:', e);
				});
		}
	});

	// Set up title update confirmation callback
	$effect(() => {
		setTitleUpdateConfirmationCallback(async (currentTitle: string, newTitle: string) => {
			return new Promise<boolean>((resolve) => {
				titleUpdateCurrentTitle = currentTitle;
				titleUpdateNewTitle = newTitle;
				titleUpdateResolve = resolve;
				titleUpdateDialogOpen = true;
			});
		});
	});
</script>

<ModeWatcher />

<Toaster richColors />

<MaximumContextAlertDialog />

<ConversationTitleUpdateDialog
	bind:open={titleUpdateDialogOpen}
	currentTitle={titleUpdateCurrentTitle}
	newTitle={titleUpdateNewTitle}
	onConfirm={handleTitleUpdateConfirm}
	onCancel={handleTitleUpdateCancel}
/>

<Sidebar.Provider bind:open={sidebarOpen}>
	<div class="flex h-screen w-full" style:height="{innerHeight}px">
		<Sidebar.Root class="h-full">
			<ChatSidebar bind:this={chatSidebar} />
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

<svelte:window onkeydown={handleKeydown} bind:innerHeight />
