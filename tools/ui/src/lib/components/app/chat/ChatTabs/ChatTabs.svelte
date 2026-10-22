<script lang="ts">
	import ChatTabsItem from './ChatTabsItem.svelte';
	import ChatTabsNewChatButton from './ChatTabsNewChatButton.svelte';
	import { ArrowRightToLine, ListX, Square, SquarePen, X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ContextMenuActions, ScrollCarousel } from '$lib/components/app';
	import {
		CHAT_TABS_MAX_WIDTH,
		NEW_CHAT_LABEL,
		NEW_CHAT_TAB_ID,
		SHORTCUT_CLOSE_TAB,
		UI_DATA_ATTRS,
		UNNAMED_CHAT_LABEL
	} from '$lib/constants';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
	import { chatStore, conversationsStore, projectsStore, tabsStore, uiStore } from '$lib/stores';
	import type { MenuAction } from '$lib/types';
	import { projectNewChatTabId, projectOfNewChatTab, resolveTabView } from '$lib/utils';
	import { type Snippet, tick } from 'svelte';

	interface Props {
		/** Sits before the tabs, e.g. the breadcrumb of a project view. */
		leading?: Snippet;
	}

	let { leading }: Props = $props();

	const carousel = useScrollCarousel();

	let view = $derived(resolveTabView(page, tabsStore.openTabs, conversationsStore.conversations));

	let activeId = $derived(view.activeId);

	let tabs = $derived(
		view.ids.map((id) =>
			id === NEW_CHAT_TAB_ID || projectOfNewChatTab(id)
				? { href: view.hrefFor(id), icon: SquarePen, id, name: NEW_CHAT_LABEL }
				: chatTab(id)
		)
	);

	// a project chat carries its project name before the title, like a breadcrumb,
	// except inside the project's own view, which leads with the breadcrumb itself
	function chatTab(id: string) {
		const conversation = conversationsStore.conversations.find((c) => c.id === id);
		const projectId = view.projectId ? undefined : conversation?.projectId;

		return {
			href: view.hrefFor(id),
			id,
			name: conversation?.name ?? UNNAMED_CHAT_LABEL,
			prefix: projectId ? projectsStore.get(projectId)?.name : undefined
		};
	}

	// the new-chat tab of the view: the regular one, or the project's own
	let newChatTabId = $derived(
		view.projectId ? projectNewChatTabId(view.projectId) : NEW_CHAT_TAB_ID
	);

	// hide the New chat button while that tab is already open
	let showNewChatButton = $derived(!view.ids.includes(newChatTabId));

	let loadingIds = $derived(new Set(chatStore.getAllLoadingChats()));

	function handleClose(id: string) {
		void tabsStore.close(id, id === activeId, view);
	}

	function openNewChat() {
		void tabsStore.openInNewTab(newChatTabId);
		void goto(view.hrefFor(newChatTabId));
	}

	function handleStop(id: string, event: MouseEvent) {
		event.stopPropagation();
		void chatStore.stopGenerationForChat(id);
	}

	function tabActions(id: string): MenuAction[] {
		const isLoading = loadingIds.has(id);
		const others = view.ids.filter((tabId) => tabId !== id);
		const toTheRight = view.ids.slice(view.ids.indexOf(id) + 1);

		return [
			...(isLoading
				? [
						{
							icon: Square,
							label: 'Stop generation',
							onclick: () => void chatStore.stopGenerationForChat(id)
						}
					]
				: []),
			{
				icon: X,
				label: 'Close tab',
				onclick: () => handleClose(id),
				separator: isLoading,
				// the shortcut closes the active tab only
				shortcut: id === activeId ? SHORTCUT_CLOSE_TAB : undefined
			},
			{
				disabled: others.length === 0,
				icon: ListX,
				label: 'Close other tabs',
				onclick: () => void tabsStore.closeMany(others, id, activeId, view)
			},
			{
				disabled: toTheRight.length === 0,
				icon: ArrowRightToLine,
				label: 'Close tabs to the right',
				onclick: () => void tabsStore.closeMany(toTheRight, id, activeId, view)
			}
		];
	}

	function handleAuxClick(id: string, event: MouseEvent) {
		// middle-click closes, like browser tabs
		if (event.button === 1) {
			event.preventDefault();
			handleClose(id);
		}
	}

	let previousTabIds = new Set<string>();
	let previousActiveId: string | null = null;

	$effect(() => {
		const currentIds = new Set(tabs.map((t) => t.id));
		const hasAddedTab = tabs.some((t) => !previousTabIds.has(t.id));

		previousTabIds = currentIds;

		const activeChanged = activeId !== previousActiveId;

		previousActiveId = activeId;

		// scroll when the active tab changes (a click) or when a new tab is added
		if (!hasAddedTab && !activeChanged) return;

		// wait for the new tab to be laid out before scrolling to it
		void tick().then(() => {
			const el = carousel.scrollContainer?.querySelector<HTMLElement>(
				`[${UI_DATA_ATTRS.ACTIVE_TAB}]`
			);

			if (el) {
				carousel.scrollToCenter(el);
			}
		});
	});
</script>

<nav
	aria-label="Open conversations"
	class={[
		'group sticky pl-1 top-0 z-10 hidden md:block chat-top-fade transition-[padding] duration-200 ease-in-out pt-3.25',
		uiStore.isSidebarExpanded
			? CHAT_TABS_MAX_WIDTH.EXPANDED_SIDEBAR
			: CHAT_TABS_MAX_WIDTH.COLLAPSED_SIDEBAR
	]}
>
	<div class="flex items-center">
		{#if leading}
			<div class="flex shrink-0 items-center pl-2">{@render leading()}</div>
		{/if}

		<div class="relative min-w-0 flex-1">
			<ScrollCarousel
				{carousel}
				class="h-10"
				containerClass="flex h-10 min-w-0 items-center"
				innerClass="items-center gap-1.25"
			>
				{#each tabs as tab (tab.id)}
					<ContextMenuActions actions={tabActions(tab.id)}>
						<ChatTabsItem
							isActive={tab.id === activeId}
							isLoading={loadingIds.has(tab.id)}
							onActivate={(id) => void goto(view.hrefFor(id))}
							onAuxClick={handleAuxClick}
							onClose={handleClose}
							onStop={handleStop}
							{tab}
						/>
					</ContextMenuActions>
				{/each}

				{#if showNewChatButton}
					<ChatTabsNewChatButton onclick={openNewChat} />
				{/if}
			</ScrollCarousel>

			<div
				class="pointer-events-none absolute inset-y-0 left-0 z-[5] w-8 bg-gradient-to-r from-background to-transparent transition-opacity {carousel.canScrollLeft
					? 'opacity-100'
					: 'opacity-0'}"
			></div>

			<div
				class="pointer-events-none absolute inset-y-0 right-0 z-[5] w-8 bg-gradient-to-l from-background to-transparent transition-opacity {carousel.canScrollRight
					? 'opacity-100'
					: 'opacity-0'}"
			></div>
		</div>
	</div>
</nav>
