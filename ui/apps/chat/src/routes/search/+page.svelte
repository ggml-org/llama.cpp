<script lang="ts">
	import { SearchInput, SidebarNavigationSearchResults } from '#lib/components/app/index.ts';
	import { ROUTES } from '#lib/constants/index.ts';
	import { RouterService } from '#lib/services/router.service.ts';
	import { chatStore, conversationsStore, deviceStore } from '#lib/stores/index.ts';
	import { browser } from '$app/env';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';

	let searchQuery = $state('');
	let searchInputRef = $state<HTMLInputElement | null>(null);

	let currentChatId = $derived(page.params.id);

	let filteredConversations = $derived.by(() => {
		const query = searchQuery.trim().toLowerCase();

		if (query.length === 0) return [];

		return conversationsStore.conversations.filter((c) => c.name.toLowerCase().includes(query));
	});

	// Search page is intended for mobile; on desktop the sidebar already exposes
	// in-place search, so bounce back to a new-chat tab without a history entry.
	$effect(() => {
		if (browser && !deviceStore.isMobile) {
			goto(ROUTES.START, { replace: true });
		}
	});

	async function selectConversation(id: string) {
		await goto(RouterService.chat(id));
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}

	function handleBack() {
		if (history.length > 1) {
			history.back();
		} else {
			conversationsStore.openNewChat();
		}
	}
</script>

<svelte:head>
	<title>Search · llama.cpp</title>
</svelte:head>

<div class="fixed top-0 z-10 left-0 right-0 p-2">
	<SearchInput
		bind:ref={searchInputRef}
		bind:value={searchQuery}
		autofocus
		onClose={handleBack}
		placeholder="Search conversations..."
	/>
</div>

<div class="p-2 pt-16">
	<SidebarNavigationSearchResults
		{currentChatId}
		{filteredConversations}
		onSelect={selectConversation}
		onStop={handleStopGeneration}
		{searchQuery}
	/>
</div>
