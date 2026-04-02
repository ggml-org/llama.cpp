<script lang="ts">
	import '../app.css';
	import { base } from '$app/paths';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import {
		ChatSidebar,
		ChatSettings,
		McpLogo,
		DialogConversationTitleUpdate,
		DialogChatSettingsImportExport
	} from '$lib/components/app';
	import { Database, Settings, Search, SquarePen } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import { goto } from '$app/navigation';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import { setImportExportDialogContext } from '$lib/contexts';

	let { children } = $props();

	let alwaysShowSidebarOnDesktop = $derived(config().alwaysShowSidebarOnDesktop);
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

	let activePanel = $state<'chat' | 'settings' | 'mcp'>('chat');
	let isMcpActive = $derived(page.route.id === '/settings/mcp');
	let isImportExportActive = $derived(page.route.id === '/settings/import-export');
	let isSettingsActive = $derived(page.route.id === '/settings/chat');
	let isSettingsRoute = $derived(!!page.route.id?.startsWith('/settings'));
	let chatSettingsRef: ChatSettings | undefined = $state();
	let importExportDialogOpen = $state(false);

	setImportExportDialogContext({
		open: () => {
			importExportDialogOpen = true;
		}
	});

	$effect(() => {
		if (activePanel === 'settings' && chatSettingsRef) {
			chatSettingsRef.reset();
		}
	});

	// Return to chat when navigating to a new route
	$effect(() => {
		void page.url;
		activePanel = 'chat';
	});

	// Global keyboard shortcuts
	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (isCtrlOrCmd && event.key === KeyboardKey.K_LOWER) {
			event.preventDefault();
			if (chatSidebar?.activateSearchMode) {
				chatSidebar.activateSearchMode();
				sidebarOpen = true;
			}
		}

		if (isCtrlOrCmd && event.shiftKey && event.key === KeyboardKey.O_UPPER) {
			event.preventDefault();
			goto('?new_chat=true#/');
		}

		if (event.shiftKey && isCtrlOrCmd && event.key === KeyboardKey.E_UPPER) {
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
		// Don't auto-open or auto-close sidebar during navigation - user controls it manually
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

	// Background MCP server health checks on app load
	// Fetch enabled servers from settings and run health checks in background
	$effect(() => {
		if (!browser) return;

		const mcpServers = mcpStore.getServers();

		// Only run health checks if we have enabled servers with URLs
		const enabledServers = mcpServers.filter((s) => s.enabled && s.url.trim());

		if (enabledServers.length > 0) {
			untrack(() => {
				// Run health checks in background (don't await)
				mcpStore.runHealthChecksForServers(enabledServers, false).catch((error) => {
					console.warn('[layout] MCP health checks failed:', error);
				});
			});
		}
	});

	// Monitor API key changes and redirect to error page if removed or changed when required
	$effect(() => {
		const apiKey = config().apiKey;

		if (
			(page.route.id === '/(chat)' || page.route.id === '/(chat)/chat/[id]') &&
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

	<DialogChatSettingsImportExport
		open={importExportDialogOpen}
		onOpenChange={(open) => (importExportDialogOpen = open)}
	/>

	<DialogConversationTitleUpdate
		bind:open={titleUpdateDialogOpen}
		currentTitle={titleUpdateCurrentTitle}
		newTitle={titleUpdateNewTitle}
		onConfirm={handleTitleUpdateConfirm}
		onCancel={handleTitleUpdateCancel}
	/>

	<Sidebar.Provider bind:open={sidebarOpen}>
		<div class="flex h-screen w-full" style:height="{innerHeight}px">
			<Sidebar.Root variant="floating" class="h-full">
				<ChatSidebar bind:this={chatSidebar} />
			</Sidebar.Root>

			{#if !(alwaysShowSidebarOnDesktop && isDesktop) && !(isSettingsRoute && !isDesktop)}
				<Sidebar.Trigger
					class="transition-left absolute left-0 z-[900] duration-200 ease-linear {sidebarOpen
						? 'left-[calc(var(--sidebar-width)+0.75rem)] max-md:hidden'
						: 'left-0!'}"
					style="translate: 1rem 1rem;"
				/>
			{/if}

			{#if isDesktop && !alwaysShowSidebarOnDesktop}
				<!-- Desktop: icon strip, always rendered, transitions width/opacity -->
				<aside
					class="hidden shrink-0 flex-col items-center justify-between overflow-hidden py-3 transition-[width,opacity] duration-200 ease-linear md:flex {sidebarOpen
						? 'pointer-events-none w-0 opacity-0'
						: 'w-[calc(var(--sidebar-width-icon)+1.5rem)] opacity-100'}"
				>
					<div class="mt-12 flex flex-col items-center gap-1">
						<Button variant="ghost" size="icon-lg" class="rounded-full" href="?new_chat=true#/">
							<SquarePen class="h-4 w-4" />
							<span class="sr-only">New Chat</span>
						</Button>
						<Button
							variant="ghost"
							size="icon-lg"
							class="rounded-full"
							onclick={() => {
								if (chatSidebar?.activateSearchMode) {
									chatSidebar.activateSearchMode();
								}
								sidebarOpen = true;
							}}
						>
							<Search class="h-4 w-4" />
							<span class="sr-only">Search</span>
						</Button>
					</div>
					<div class="flex flex-col items-center gap-1">
						<Button
							variant="ghost"
							size="icon-lg"
							href="#/settings/mcp"
							class="rounded-full {isMcpActive ? 'bg-accent text-accent-foreground' : ''}"
						>
							<McpLogo class="h-4 w-4" />
							<span class="sr-only">MCP Servers</span>
						</Button>
						<Button
							variant="ghost"
							size="icon-lg"
							href="#/settings/import-export"
							class="rounded-full {isImportExportActive ? 'bg-accent text-accent-foreground' : ''}"
						>
							<Database class="h-4 w-4" />
							<span class="sr-only">Import / Export</span>
						</Button>
						<Button
							variant="ghost"
							size="icon-lg"
							href="#/settings/chat"
							class="rounded-full {isSettingsActive ? 'bg-accent text-accent-foreground' : ''}"
						>
							<Settings class="h-4 w-4" />
							<span class="sr-only">Settings</span>
						</Button>
					</div>
				</aside>
			{/if}

			<Sidebar.Inset class="flex flex-1 flex-col overflow-auto">
				{@render children?.()}
			</Sidebar.Inset>
		</div>
	</Sidebar.Provider>
</Tooltip.Provider>

<svelte:window onkeydown={handleKeydown} bind:innerHeight />
