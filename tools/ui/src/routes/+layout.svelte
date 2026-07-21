<script lang="ts">
	import '../app.css';
	import { base } from '$app/paths';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import { onMount } from 'svelte';

	import { SidebarNavigation } from '$lib/components/app';
	import { PwaMetaTags, PwaRefreshAlert } from '$lib/components/pwa';
	import { pwaAssetsHead } from 'virtual:pwa-assets/head';

	import { chatStore } from '$lib/stores/chat.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { ROUTES } from '$lib/constants/routes';
	import { RouterService } from '$lib/services/router.service';
	import { Toaster } from 'svelte-sonner';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { AUTHORIZATION_HEADER, BEARER_PREFIX, TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { FAVICON_PATHS, FAVICON_SELECTORS } from '$lib/constants/pwa';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { usePwa } from '$lib/hooks/use-pwa.svelte';
	import { conversations } from '$lib/stores/conversations.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { theme } from '$lib/stores/theme.svelte';
	import { buildInfoStore } from '$lib/stores/build-info.svelte';

	import { SETTINGS_KEYS } from '$lib/constants';

	let { children } = $props();
	let innerHeight = $state<number | undefined>();
	let innerWidth = $state(browser ? window.innerWidth : 0);

	let chatSidebar:
		| {
				activateSearchMode?: () => void;
				editActiveConversation?: () => void;
		  }
		| undefined = $state();

	let showBuildVersion = $derived(config()[SETTINGS_KEYS.SHOW_BUILD_VERSION] as boolean);

	// Keep the hook object intact: destructuring needRefreshByStorage reads the getter once and freezes it
	const pwa = usePwa();
	const { needRefresh, updateServiceWorker } = pwa;

	function updateFavicon() {
		const dark = theme.isSystemDark;

		let icoLink = document.querySelector(FAVICON_SELECTORS.ICO_48X48) as HTMLLinkElement | null;
		if (icoLink) {
			icoLink.href = dark ? FAVICON_PATHS.ICO_DARK : FAVICON_PATHS.ICO_LIGHT;
		}

		let svgLink = document.querySelector(FAVICON_SELECTORS.SVG_ANY) as HTMLLinkElement | null;
		if (svgLink) {
			svgLink.href = dark ? FAVICON_PATHS.SVG_DARK : FAVICON_PATHS.SVG_LIGHT;
		}
	}

	function navigateToConversation(direction: -1 | 1) {
		const allConvs = conversations();

		if (allConvs.length === 0) return;

		const currentId = page.params.id;

		if (!currentId) {
			goto(RouterService.chat(allConvs[direction === 1 ? 0 : allConvs.length - 1].id));

			return;
		}

		const idx = allConvs.findIndex((c) => c.id === currentId);

		if (idx === -1) return;

		const targetIdx = idx + direction;

		if (targetIdx >= 0 && targetIdx < allConvs.length) {
			goto(RouterService.chat(allConvs[targetIdx].id));
		} else {
			goto(ROUTES.NEW_CHAT);
		}
	}

	// Global keyboard shortcuts
	const { handleKeydown } = useKeyboardShortcuts({
		editActiveConversation: () => chatSidebar?.editActiveConversation?.(),
		navigateToPrevConversation: () => navigateToConversation(-1),
		navigateToNextConversation: () => navigateToConversation(1)
	});

	function checkApiKey() {
		const apiKey = config().apiKey;

		// No API key configured — server doesn't require auth, no need to validate.
		// This mirrors the early return in validateApiKey() to avoid redundant /props requests.
		if (!apiKey || apiKey.trim() === '') {
			return;
		}

		untrack(() => {
			if (
				(page.route.id === '/(chat)' || page.route.id === '/(chat)/chat/[id]') &&
				page.status !== 401 &&
				page.status !== 403
			) {
				const headers: Record<string, string> = {
					'Content-Type': 'application/json',
					[AUTHORIZATION_HEADER]: `${BEARER_PREFIX}${apiKey.trim()}`
				};

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
	}

	onMount(() => {
		updateFavicon();
		// snapshot of every backend running stream on first load, populates the sidebar spinners
		// so the user sees each conv that has a live inference, even ones not opened yet
		void chatStore.syncRemoteRunningStreams();
	});

	// refresh that snapshot when the tab returns to the foreground, a stream may have advanced
	// or ended while it was hidden. snapshot only, no polling
	function handleVisibilityChange() {
		if (document.visibilityState !== 'visible') return;
		void chatStore.syncRemoteRunningStreams();
	}

	$effect(() => {
		void theme.isSystemDark;

		updateFavicon();
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
			untrack(() => {
				settingsStore.syncWithServerDefaults();
			});
		}
	});

	// Inject custom CSS at runtime through an action on the head style node
	// textContent keeps the value as text, never parsed as HTML
	function customCss(node: HTMLStyleElement) {
		$effect(() => {
			node.textContent = (config().customCss as string | undefined) ?? '';
		});
	}

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

	// Live model status and load progress via the /models/sse feed (router mode)
	$effect(() => {
		if (!browser) return;
		if (!isRouterMode()) return;

		untrack(() => {
			modelsStore.subscribeStatus();
		});

		return () => {
			modelsStore.unsubscribeStatus();
		};
	});

	// Background MCP server health checks on app load.
	// Health-check every configured server with a URL - including disabled ones -
	// so the /mcp-servers page can display health metadata for servers that are
	// currently turned off. Disabled servers never get promoted to active
	// connections (see runHealthCheck), so their tools/prompts/resources stay
	// out of the chat-side stores.
	// Only IDLE servers are checked; already-resolved (SUCCESS / ERROR) servers
	// keep their existing state, so adding or removing a server does not flash
	// every other card back through skeleton state.
	$effect(() => {
		if (!browser) return;

		const mcpServers = mcpStore.getServers();

		const serversWithUrls = mcpServers.filter((s) => s.url.trim());

		if (serversWithUrls.length > 0) {
			untrack(() => {
				// Run health checks in background (don't await)
				mcpStore.runHealthChecksForServers(serversWithUrls, true).catch((error) => {
					console.warn('[layout] MCP health checks failed:', error);
				});
			});
		}
	});

	// Monitor API key changes and redirect to error page if removed or changed when required
	$effect(() => {
		checkApiKey();
	});
</script>

<svelte:head>
	{#if pwaAssetsHead.themeColor}
		<meta name="theme-color" content={pwaAssetsHead.themeColor.content} />
	{/if}

	{#if config().customCss}
		<style use:customCss></style>
	{/if}

	{#each pwaAssetsHead.links as link (link.href)}
		<link {...link} />
	{/each}

	<PwaMetaTags />
</svelte:head>

<svelte:window onkeydown={handleKeydown} bind:innerHeight bind:innerWidth />
<svelte:document onvisibilitychange={handleVisibilityChange} />

<Tooltip.Provider delayDuration={TOOLTIP_DELAY_DURATION}>
	<div class="flex flex-col md:flex-row">
		<SidebarNavigation
			onSearchClick={() => {
				if (isMobile.current) {
					goto(ROUTES.SEARCH);
				} else if (chatSidebar?.activateSearchMode) {
					chatSidebar.activateSearchMode();
				}
			}}
		/>

		<div class="flex-1">
			{@render children?.()}
		</div>
	</div>

	<ModeWatcher />

	<Toaster richColors />
</Tooltip.Provider>

<!-- PWA update prompt + version -->
<div class="fixed right-4 bottom-4 z-9999 flex flex-col items-end gap-1">
	{#if showBuildVersion && buildInfoStore.value}
		<span class="text-[10px] tabular-nums text-muted-foreground">{buildInfoStore.value}</span>
	{/if}

	<PwaRefreshAlert
		needRefresh={$needRefresh || pwa.needRefreshByStorage}
		forceReload={pwa.needRefreshByStorage}
		{updateServiceWorker}
	/>
</div>
