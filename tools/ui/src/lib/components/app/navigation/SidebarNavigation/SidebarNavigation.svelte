<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { PanelLeftClose } from '@lucide/svelte';
	import {
		ActionIcon,
		Logo,
		SidebarNavigationConversationList,
		SidebarNavigationActions
	} from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { fade } from 'svelte/transition';
	import { circIn } from 'svelte/easing';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { RouterService } from '$lib/services/router.service';

	interface Props {
		onSearchClick?: () => void;
	}

	let { onSearchClick = () => {} }: Props = $props();

	const { handleKeydown } = useKeyboardShortcuts({ activateSearchMode: () => onSearchClick() });

	let isExpandedMode = $state(false);
	let hoveredTooltip = $state<string | null>(null);

	const isStripExpanded = $derived(isExpandedMode || hoveredTooltip !== null);

	function toggleExpandedMode() {
		isExpandedMode = !isExpandedMode;
		if (!isExpandedMode) {
			hoveredTooltip = null;
		}
	}

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');

	let filteredConversations = $derived.by(() => {
		if (isSearchModeActive) {
			if (searchQuery.trim().length > 0) {
				return conversations().filter((conversation: { name: string }) =>
					conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
				);
			}

			return [];
		}

		return conversations();
	});

	async function selectConversation(id: string) {
		await goto(RouterService.chat(id));
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (!conversation) return;

		const newName = window.prompt('Rename conversation', conversation.name);
		if (newName && newName.trim()) {
			await conversationsStore.updateConversationName(id, newName.trim());
		}
	}

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
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
</script>

<svelte:window onkeydown={handleKeydown} />

<aside
	class="sticky top-2 left-2 max-h-[calc(100dvh-1rem)] py-2 rounded-xl z-10 hidden flex-col justify-between transition-[width,padding] duration-200 ease-out md:flex {isStripExpanded
		? 'w-72 bg-muted/50 backdrop-blur-xl border-border shadow-md'
		: 'w-12'}"
>
	<div class="px-2 flex items-center justify-between">
		<ActionIcon
			icon={Logo}
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent!"
			href={isExpandedMode ? ROUTES.START : undefined}
			onclick={isExpandedMode ? undefined : toggleExpandedMode}
			ariaLabel={isExpandedMode ? 'Go to start' : 'Expand navigation'}
		/>

		{#if isExpandedMode}
			<div in:fade={{ duration: 150, easing: circIn, delay: 50 }} out:fade={{ duration: 100 }}>
				<ActionIcon
					icon={PanelLeftClose}
					size="lg"
					iconSize="h-4 w-4"
					class="h-9 w-9 rounded-full mr-1 hover:bg-accent!"
					onclick={toggleExpandedMode}
					ariaLabel="Collapse navigation"
				/>
			</div>
		{/if}
	</div>

	<div class="mt-2 flex min-h-0 flex-1 flex-col gap-1">
		<SidebarNavigationActions {isExpandedMode} class="px-2" />

		{#if isExpandedMode}
			<div
				class="flex min-h-0 flex-1 flex-col"
				in:fade={{ duration: 150, easing: circIn, delay: 50 }}
				out:fade={{ duration: 100 }}
			>
				<SidebarNavigationConversationList
					class="px-2"
					{filteredConversations}
					{currentChatId}
					{isSearchModeActive}
					{searchQuery}
					onSelect={selectConversation}
					onEdit={handleEditConversation}
					onDelete={handleDeleteConversation}
					onStop={handleStopGeneration}
				/>
			</div>
		{/if}
	</div>
</aside>
