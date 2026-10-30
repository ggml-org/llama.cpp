<script lang="ts">
	import '../app.css';
	import { browser } from '$app/environment';
	import { afterNavigate, goto } from '$app/navigation';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { ActionIcon, MenuIcon, SidebarNavigation } from '$lib/components/app';
	import { PwaMetaTags, PwaRefreshAlert } from '$lib/components/pwa';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		BOX_BORDER,
		FAVICON_PATHS,
		FAVICON_SELECTORS,
		HEADERS,
		MODELS_ROUTE_ID,
		NEW_CHAT_TAB_ID,
		ROUTES,
		SETTINGS_KEYS,
		TOOLTIP_DELAY_DURATION
	} from '$lib/constants';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { usePwa } from '$lib/hooks/use-pwa.svelte';
	import { useSidebarReveal } from '$lib/hooks/use-sidebar-reveal.svelte';
	import { RouterService } from '$lib/services/router.service';
	import {
		chatStore,
		conversationsStore,
		deviceStore,
		mcpStore,
		modelsStore,
		serverStore,
		settingsStore,
		tabsStore,
		uiStore,
		versionStore
	} from '$lib/stores';
	import { initStores } from '$lib/stores/init';
	import { ModeWatcher } from 'mode-watcher';
	import { untrack } from 'svelte';
	import { onMount } from 'svelte';
	import { Toaster } from 'svelte-sonner';
	import { pwaAssetsHead } from 'virtual:pwa-assets/head';

	let { children } = $props();

	// migrations and store startup, ordered explicitly instead of import side effects
	void initStores();

	let innerHeight = $state<number | undefined>();
	let innerWidth = $state(browser ? window.innerWidth : 0);

	let chatSidebar: { editActiveConversation?: () => void } | undefined = $state();

	let showBuildVersion = $derived(
		settingsStore.config[SETTINGS_KEYS.SHOW_BUILD_VERSION] as boolean
	);

	// Keep the hook object intact: destructuring needRefreshByStorage reads the getter once and freezes it
	const pwa = usePwa();
	const { needRefresh, updateServiceWorker } = pwa;

	function updateFavicon() {
		const dark = deviceStore.systemTheme.isDark;

		let icoLink = document.querySelector(FAVICON_SELECTORS.ICO_48X48) as HTMLLinkElement | null;

		if (icoLink) {
			icoLink.href = dark ? FAVICON_PATHS.ICO_DARK : FAVICON_PATHS.ICO_LIGHT;
		}

		let svgLink = document.querySelector(FAVICON_SELECTORS.SVG_ANY) as HTMLLinkElement | null;

		if (svgLink) {
			svgLink.href = dark ? FAVICON_PATHS.SVG_DARK : FAVICON_PATHS.SVG_LIGHT;
		}
	}

	function navigateToTab(direction: -1 | 1) {
		// only makes sense with conversation tabs enabled
		if (!settingsStore.config.conversationTabs) return;

		const openTabs = tabsStore.openTabs;

		if (openTabs.length === 0) return;

		const activeId = page.params.id ?? NEW_CHAT_TAB_ID;
		const idx = openTabs.indexOf(activeId);
		// active tab not in list (e.g. a non-chat route): start from an edge
		const targetIdx =
			idx === -1
				? direction === 1
					? 0
					: openTabs.length - 1
				: (idx + direction + openTabs.length) % openTabs.length;

		void tabsStore.activate(openTabs[targetIdx]);
	}

	function navigateToConversation(direction: -1 | 1) {
		const allConvs = conversationsStore.conversations;

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
			conversationsStore.openNewChat();
		}
	}

	const sidebarReveal = useSidebarReveal();

	const isRevealed = $derived(uiStore.isSidebarExpanded || sidebarReveal.dragOffset !== null);

	// phone only: the page slides aside by the sidebar's width, or follows the finger mid-drag
	const revealTransform = $derived.by(() => {
		if (!deviceStore.isMobile) return undefined;

		if (sidebarReveal.dragOffset !== null)
			return `translate3d(${sidebarReveal.dragOffset}px, 0, 0)`;

		return uiStore.isSidebarExpanded ? 'translateX(var(--sidebar-reveal))' : undefined;
	});

	// the models page returns to wherever it was reached from, by any kind of navigation
	afterNavigate(() => uiStore.rememberModelsReturnRoute());

	// Global keyboard shortcuts
	const { handleKeydown } = useKeyboardShortcuts({
		editActiveConversation: () => chatSidebar?.editActiveConversation?.(),
		navigateToNextConversation: () => navigateToConversation(1),
		navigateToNextTab: () => navigateToTab(1),
		navigateToPrevConversation: () => navigateToConversation(-1),
		navigateToPrevTab: () => navigateToTab(-1)
	});

	function checkApiKey() {
		const apiKey = settingsStore.config.apiKey;

		// Without a stored key there is nothing to re-validate here; the keyless
		// 401 case is handled by validateApiKey() at navigation time, and the
		// reload below must never fire in a keyless loop.
		if (!apiKey || apiKey.trim() === '') {
			return;
		}

		untrack(() => {
			if (
				(page.route.id === '/(chat)' ||
					page.route.id === '/(chat)/chat/[id]' ||
					page.route.id === MODELS_ROUTE_ID) &&
				page.status !== 401 &&
				page.status !== 403
			) {
				const headers: Record<string, string> = {
					'Content-Type': 'application/json',
					[HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${apiKey.trim()}`
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
		void deviceStore.systemTheme.isDark;

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
			node.textContent = (settingsStore.config.customCss as string | undefined) ?? '';
		});
	}

	// Fetch router models when in router mode (for status and modalities)
	// Wait for models to be loaded first, run only once
	let routerModelsFetched = false;

	$effect(() => {
		const isRouter = serverStore.isRouterMode;
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

		if (!serverStore.isRouterMode) return;

		untrack(() => {
			modelsStore.status.subscribe();
		});

		return () => {
			modelsStore.status.unsubscribe();
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
		<meta content={pwaAssetsHead.themeColor.content} name="theme-color" />
	{/if}

	{#if settingsStore.config.customCss}
		<style use:customCss></style>
	{/if}

	{#each pwaAssetsHead.links as link (link.href)}
		<link {...link} />
	{/each}

	<PwaMetaTags />
</svelte:head>

<svelte:window bind:innerHeight bind:innerWidth onkeydown={handleKeydown} />
<svelte:document onvisibilitychange={handleVisibilityChange} />

<Tooltip.Provider delayDuration={TOOLTIP_DELAY_DURATION}>
	<!-- clip, not hide: the page slid aside for the phone sidebar must not widen the document,
		 or iOS pans it sideways and leaves it stuck there -->
	<div class="flex flex-col overflow-x-clip md:flex-row">
		<SidebarNavigation />

		<!-- min-w-0 lets the chat column shrink below its content width, so wide
		     code blocks and tables scroll inside their own containers instead of
		     stretching the page into a horizontal scrollbar -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			style:transform={revealTransform}
			class={[
				'min-w-0 flex-1',
				deviceStore.isMobile && 'relative z-10 bg-background',
				// the installed iOS app sizes the locked, slid page to the screen less its status bar,
				// so it reaches down by that much more; locked, it cannot scroll by it
				deviceStore.isMobile &&
					(sidebarReveal.isRevealed ? 'min-h-[calc(100dvh+var(--safe-area-top))]' : 'min-h-dvh'),
				deviceStore.isMobile &&
					sidebarReveal.dragOffset === null &&
					'transition-transform duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]',
				// slid aside: nothing inside scrolls or takes a tap, which lands on the page itself and
				// closes the sidebar; forced on every descendant, as the fixed bars opt back into events
				sidebarReveal.isRevealed && 'touch-none [&_*]:pointer-events-none!'
			]}
			onclickcapture={sidebarReveal.handleClickCapture}
			onpointercancel={sidebarReveal.handlePointerUp}
			onpointerdown={sidebarReveal.handlePointerDown}
			onpointermove={sidebarReveal.handlePointerMove}
			onpointerup={sidebarReveal.handlePointerUp}
		>
			<!-- the phone sidebar sits under the page, so its toggle rides on the page -->
			{#if deviceStore.isMobile && !page.url.hash.includes(ROUTES.SEARCH)}
				<ActionIcon
					ariaLabel={uiStore.isSidebarExpanded ? 'Collapse navigation' : 'Expand navigation'}
					class="sidebar-reveal-pin-top fixed top-(--corner-control-top) left-4 z-20 transition-colors h-10 w-10 rounded-full bg-muted/60 border border-border/20 shadow-sm text-accent-foreground"
					icon={MenuIcon}
					iconSize="size-6"
					onclick={() => (uiStore.isSidebarExpanded = !uiStore.isSidebarExpanded)}
					size="lg"
				/>
			{/if}

			{@render children?.()}
		</div>

		<!-- frames the slid page as a rounded card; its corners are masked in the sidebar tint -->
		{#if deviceStore.isMobile}
			<div
				style:transform={revealTransform}
				aria-hidden="true"
				class={[
					'pointer-events-none fixed inset-0 z-20 overflow-hidden',
					sidebarReveal.dragOffset === null &&
						'transition-[transform,opacity] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]',
					// on close the corners hold until the page is nearly home, then fade
					isRevealed
						? 'opacity-100'
						: 'opacity-0 [transition-delay:0ms,150ms] [transition-duration:300ms,150ms]'
				]}
			>
				<div
					class="absolute inset-0 rounded-[2.5rem] shadow-[0_0_0_2.5rem_var(--sidebar-reveal-bg)] {BOX_BORDER}"
				></div>
			</div>
		{/if}
	</div>

	<ModeWatcher />

	<Toaster closeButton richColors />

	<!-- loaded on demand, so it stays out of the production bundle -->
	{#if import.meta.env.DEV && import.meta.env.VITE_DEBUG}
		{#await import('$lib/components/app/misc/ViewportProbe.svelte') then { default: ViewportProbe }}
			<ViewportProbe />
		{/await}
	{/if}
</Tooltip.Provider>

<!-- PWA update prompt + version -->
<div class="fixed right-4 bottom-4 z-9999 flex flex-col items-end gap-1">
	{#if showBuildVersion && versionStore.build}
		<span class="text-[10px] tabular-nums text-muted-foreground">{versionStore.build}</span>
	{/if}

	<PwaRefreshAlert
		forceReload={pwa.needRefreshByStorage}
		needRefresh={$needRefresh || pwa.needRefreshByStorage}
		{updateServiceWorker}
	/>
</div>
