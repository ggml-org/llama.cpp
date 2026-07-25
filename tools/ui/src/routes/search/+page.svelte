<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { browser } from '$app/environment';

	import { SearchInput, SidebarNavigationSearchResults } from '$lib/components/app';
	import { ROUTES } from '$lib/constants/routes';
	import { RouterService } from '$lib/services/router.service';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';

	let searchQuery = $state('');
	let searchInputRef = $state<HTMLInputElement | null>(null);

	let currentChatId = $derived(page.params.id);

	let filteredConversations = $derived.by(() => {
		const query = searchQuery.trim().toLowerCase();
		if (query.length === 0) return [];
		return conversations().filter((c) => c.name.toLowerCase().includes(query));
	});

	// Search page is intended for mobile; on desktop the sidebar already exposes
	// in-place search, so bounce back to a chat.
	$effect(() => {
		if (browser && !isMobile.current) {
			goto(ROUTES.NEW_CHAT, { replaceState: true });
		}
	});

	async function selectConversation(id: string) {
		await goto(RouterService.chat(id));
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((c) => c.id === id);
		if (!conversation) return;

		const newName = window.prompt('Rename conversation', conversation.name);
		if (newName && newName.trim()) {
			await conversationsStore.updateConversationName(id, newName.trim());
		}
	}

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((c) => c.id === id);
		if (!conversation) return;

		const confirmed = window.confirm(
			`Delete "${conversation.name}"? This action cannot be undone.`
		);
		if (!confirmed) return;

		await conversationsStore.deleteConversation(id, { deleteWithForks: false });
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}

	function handleBack() {
		if (history.length > 1) {
			history.back();
		} else {
			goto(ROUTES.NEW_CHAT);
		}
	}
</script>

<svelte:head>
	<title>Search · llama.cpp</title>
</svelte:head>

<div class="fixed top-0 z-10 left-0 right-0 p-2">
	<SearchInput
		autofocus
		bind:value={searchQuery}
		bind:ref={searchInputRef}
		onClose={handleBack}
		placeholder="Search conversations..."
	/>
</div>

<div class="p-2 pt-16">
	<SidebarNavigationSearchResults
		{searchQuery}
		{filteredConversations}
		{currentChatId}
		onSelect={selectConversation}
		onEdit={handleEditConversation}
		onDelete={handleDeleteConversation}
		onStop={handleStopGeneration}
	/>
</div>
