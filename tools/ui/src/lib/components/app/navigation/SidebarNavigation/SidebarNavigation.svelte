<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { Trash2, Pencil, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { DialogConfirmation, Logo } from '$lib/components/app';
	import SidebarNavigationActions from './SidebarNavigationActions.svelte';
	import SidebarNavigationConversationList from './SidebarNavigationConversationList.svelte';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import Label from '$lib/components/ui/label/label.svelte';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import Input from '$lib/components/ui/input/input.svelte';
	import { ROUTES } from '$lib/constants/routes';
	import { RouterService } from '$lib/services/router.service';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { getPreviewText } from '$lib/utils';
	import { APP_NAME } from '$lib/constants';

	const sidebar = Sidebar.useSidebar();

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');
	let showDeleteDialog = $state(false);
	let deleteWithForks = $state(false);
	let showEditDialog = $state(false);
	let selectedConversation = $state<DatabaseConversation | null>(null);
	let editedName = $state('');
	let selectedConversationNamePreview = $derived.by(() =>
		selectedConversation ? getPreviewText(selectedConversation.name) : ''
	);

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

	let selectedConversationHasDescendants = $derived.by(() => {
		if (!selectedConversation) return false;

		const allConvs = conversations();
		const queue = [selectedConversation.id];

		while (queue.length > 0) {
			const parentId = queue.pop()!;

			for (const c of allConvs) {
				if (c.forkedFromConversationId === parentId) return true;
			}
		}

		return false;
	});

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
			deleteWithForks = false;
			showDeleteDialog = true;
		}
	}

	async function handleEditConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
			editedName = conversation.name;
			showEditDialog = true;
		}
	}

	function handleConfirmDelete() {
		if (selectedConversation) {
			const convId = selectedConversation.id;
			const withForks = deleteWithForks;
			showDeleteDialog = false;

			setTimeout(() => {
				conversationsStore.deleteConversation(convId, {
					deleteWithForks: withForks
				});
			}, 100); // Wait for animation to finish
		}
	}

	function handleConfirmEdit() {
		if (!editedName.trim() || !selectedConversation) return;

		showEditDialog = false;

		conversationsStore.updateConversationName(selectedConversation.id, editedName);
		selectedConversation = null;
	}

	export function handleMobileSidebarItemClick() {
		if (sidebar.isMobile) {
			sidebar.toggle();
		}
	}

	let chatSidebarActions: { activateSearch?: () => void } | undefined = $state();
	let openedForSearch = $state(false);

	export function activateSearchMode() {
		if (!sidebar.open) {
			openedForSearch = true;
		}
		chatSidebarActions?.activateSearch?.();
	}

	function handleSearchDeactivated() {
		if (openedForSearch) {
			openedForSearch = false;
			sidebar.toggle();
		}
	}

	$effect(() => {
		if (!sidebar.open) {
			isSearchModeActive = false;
			searchQuery = '';
			openedForSearch = false;
		}
	});

	export function editActiveConversation() {
		if (currentChatId) {
			const activeConversation = filteredConversations.find((conv) => conv.id === currentChatId);

			if (activeConversation) {
				const event = new CustomEvent('edit-active-conversation', {
					detail: { conversationId: currentChatId }
				});
				document.dispatchEvent(event);
			}
		}
	}

	async function selectConversation(id: string) {
		if (isSearchModeActive) {
			isSearchModeActive = false;
			searchQuery = '';
		}

		handleMobileSidebarItemClick();
		await goto(RouterService.chat(id));
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}
</script>

<div class="flex h-full flex-col">
	<ScrollArea class="h-full flex-1">
		<Sidebar.Header class="gap-4 bg-sidebar/50 p-3 backdrop-blur-lg md:pt-4 md:pb-2">
			<div class="flex items-center justify-between">
				<a href={ROUTES.START} onclick={handleMobileSidebarItemClick}>
					<!-- <h1 class="inline-flex items-center gap-1 px-2 text-xl font-semibold">
						{APP_NAME}
					</h1> -->
					<Logo class="sticky top-4 ml-2 mt-1.25" --size="1.125rem" />
				</a>

				<Button
					class="rounded-full md:hidden"
					variant="ghost"
					size="icon"
					onclick={() => sidebar.toggle()}
				>
					<X class="h-4 w-4" />
					<span class="sr-only">Close sidebar</span>
				</Button>
			</div>

			<SidebarNavigationActions
				bind:this={chatSidebarActions}
				{handleMobileSidebarItemClick}
				bind:isSearchModeActive
				bind:searchQuery
				onSearchDeactivated={handleSearchDeactivated}
			/>
		</Sidebar.Header>

		<SidebarNavigationConversationList
			{filteredConversations}
			{currentChatId}
			{isSearchModeActive}
			{searchQuery}
			onSelect={selectConversation}
			onEdit={handleEditConversation}
			onDelete={handleDeleteConversation}
			onStop={handleStopGeneration}
		/>
	</ScrollArea>
</div>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete Conversation"
	description={selectedConversation
		? `Are you sure you want to delete "${selectedConversationNamePreview}"? This action cannot be undone and will permanently remove all messages in this conversation.`
		: ''}
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => {
		showDeleteDialog = false;
		selectedConversation = null;
	}}
>
	{#if selectedConversationHasDescendants}
		<div class="flex items-center gap-2 py-2">
			<Checkbox id="delete-with-forks" bind:checked={deleteWithForks} />

			<Label for="delete-with-forks" class="text-sm">Also delete all forked conversations</Label>
		</div>
	{/if}
</DialogConfirmation>

<DialogConfirmation
	bind:open={showEditDialog}
	title="Edit Conversation Name"
	description=""
	confirmText="Save"
	cancelText="Cancel"
	icon={Pencil}
	onConfirm={handleConfirmEdit}
	onCancel={() => {
		showEditDialog = false;
		selectedConversation = null;
	}}
	onKeydown={(event) => {
		if (event.key === 'Enter') {
			event.preventDefault();
			event.stopImmediatePropagation();
			handleConfirmEdit();
		}
	}}
>
	<Input
		class="text-foreground"
		placeholder="Enter a new name"
		type="text"
		bind:value={editedName}
	/>
</DialogConfirmation>
