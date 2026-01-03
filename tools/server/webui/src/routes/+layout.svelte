<script lang="ts">
	import '../app.css';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import { ChatSidebar, DialogConversationTitleUpdate } from '$lib/components/app';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeMessages } from '$lib/stores/conversations.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import { goto } from '$app/navigation';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants/tooltip-config';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';

	let { children } = $props();

	let isChatRoute = $derived(page.route.id === '/chat/[id]');
	let isHomeRoute = $derived(page.route.id === '/');
	let isNewChatMode = $derived(page.url.searchParams.get('new_chat') === 'true');
	let showSidebarByDefault = $derived(activeMessages().length > 0 || isLoading());
	let alwaysShowSidebarOnDesktop = $derived(config().alwaysShowSidebarOnDesktop);
	let autoShowSidebarOnNewChat = $derived(config().autoShowSidebarOnNewChat);
	let isMobile = new IsMobile();
	let isDesktop = $derived(!isMobile.current);
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

		if (isCtrlOrCmd && event.shiftKey && event.key === 'O') {
			event.preventDefault();
			goto('?new_chat=true#/');
		}

		if (event.shiftKey && isCtrlOrCmd && event.key === 'E') {
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
		if (alwaysShowSidebarOnDesktop && isDesktop) {
			sidebarOpen = true;
			return;
		}

		if (isHomeRoute && !isNewChatMode) {
			// Auto-collapse sidebar when navigating to home route (but not in new chat mode)
			sidebarOpen = false;
		} else if (isHomeRoute && isNewChatMode) {
			// Keep sidebar open in new chat mode
			sidebarOpen = true;
		} else if (isChatRoute) {
			// On chat routes, only auto-show sidebar if setting is enabled
			if (autoShowSidebarOnNewChat) {
				sidebarOpen = true;
			}
			// If setting is disabled, don't change sidebar state - let user control it manually
		} else {
			// Other routes follow default behavior
			sidebarOpen = showSidebarByDefault;
		}
	});

	// Initialize server properties on app load (run once)
	$effect(() => {
		// Only fetch if we don't already have props
		if (!serverStore.props) {
			untrack(() => {
				serverStore.fetch();
			});
		}
	});

	// Sync settings when server props are loaded
	$effect(() => {
		const serverProps = serverStore.props;

		if (serverProps) {
			settingsStore.syncWithServerDefaults();
		}
	});

	// Fetch router models when in router mode (for status and modalities)
	// Wait for models to be loaded first, run only once
	let routerModelsFetched = false;

	$effect(() => {
		const isRouter = isRouterMode();
		const modelsCount = modelsStore.models.length;

		// Only fetch router models once when we have models loaded and in router mode
		if (isRouter && modelsCount > 0 && !routerModelsFetched) {
			routerModelsFetched = true;
			untrack(() => {
				modelsStore.fetchRouterModels();
			});
		}
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

			fetch(`${base}/props`, { headers })
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
		conversationsStore.setTitleUpdateConfirmationCallback(
			async (currentTitle: string, newTitle: string) => {
				return new Promise<boolean>((resolve) => {
					titleUpdateCurrentTitle = currentTitle;
					titleUpdateNewTitle = newTitle;
					titleUpdateResolve = resolve;
					titleUpdateDialogOpen = true;
				});
			}
		);
	});
</script>

<Tooltip.Provider delayDuration={TOOLTIP_DELAY_DURATION}>
	<ModeWatcher />

	<Toaster richColors />

	<DialogConversationTitleUpdate
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

			{#if !(alwaysShowSidebarOnDesktop && isDesktop)}
				<Sidebar.Trigger
					class="transition-left absolute left-0 z-[900] h-8 w-8 duration-200 ease-linear {sidebarOpen
						? 'md:left-[var(--sidebar-width)]'
						: ''}"
					style="translate: 1rem 1rem;"
				/>
			{/if}

			<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
				{@render children?.()}
			</Sidebar.Inset>
		</div>
	</Sidebar.Provider>
</Tooltip.Provider>

<svelte:window onkeydown={handleKeydown} bind:innerHeight />
