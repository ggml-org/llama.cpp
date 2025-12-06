<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { Trash2 } from '@lucide/svelte';
	import { ChatSidebarConversationItem, DialogConfirmation } from '$lib/components/app';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import Input from '$lib/components/ui/input/input.svelte';
	import { conversationsStore, conversations } from '$lib/stores/conversations.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import ChatSidebarActions from './ChatSidebarActions.svelte';

	const sidebar = Sidebar.useSidebar();

	let currentChatId = $derived(page.params.id);
	let isSearchModeActive = $state(false);
	let searchQuery = $state('');
	let showDeleteDialog = $state(false);
	let showEditDialog = $state(false);
	let selectedConversation = $state<DatabaseConversation | null>(null);
	let editedName = $state('');

	let filteredConversations = $derived.by(() => {
		if (searchQuery.trim().length > 0) {
			return conversations().filter((conversation: { name: string }) =>
				conversation.name.toLowerCase().includes(searchQuery.toLowerCase())
			);
		}

		return conversations();
	});

	async function handleDeleteConversation(id: string) {
		const conversation = conversations().find((conv) => conv.id === id);
		if (conversation) {
			selectedConversation = conversation;
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
			showDeleteDialog = false;

			setTimeout(() => {
				conversationsStore.deleteConversation(selectedConversation.id);
				selectedConversation = null;
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

	export function activateSearchMode() {
		isSearchModeActive = true;
	}

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

		await goto(`#/chat/${id}`);
	}

	function handleStopGeneration(id: string) {
		chatStore.stopGenerationForChat(id);
	}
</script>

<ScrollArea class="h-[100vh]">
	<Sidebar.Header class=" top-0 z-10 gap-6 bg-sidebar/50 px-4 py-4 pb-2 backdrop-blur-lg md:sticky">
		<a href="#/" onclick={handleMobileSidebarItemClick}>
			<h1 class="inline-flex items-center gap-1 px-2 text-xl font-semibold">llama.cpp</h1>
		</a>

		<ChatSidebarActions {handleMobileSidebarItemClick} bind:isSearchModeActive bind:searchQuery />
	</Sidebar.Header>

	<Sidebar.Group class="mt-4 space-y-2 p-0 px-4">
		{#if (filteredConversations.length > 0 && isSearchModeActive) || !isSearchModeActive}
			<Sidebar.GroupLabel>
				{isSearchModeActive ? 'Search results' : 'Conversations'}
			</Sidebar.GroupLabel>
		{/if}

		<Sidebar.GroupContent>
			<Sidebar.Menu>
				{#each filteredConversations as conversation (conversation.id)}
					<Sidebar.MenuItem class="mb-1">
						<ChatSidebarConversationItem
							conversation={{
								id: conversation.id,
								name: conversation.name,
								lastModified: conversation.lastModified,
								currNode: conversation.currNode
							}}
							{handleMobileSidebarItemClick}
							isActive={currentChatId === conversation.id}
							onSelect={selectConversation}
							onEdit={handleEditConversation}
							onDelete={handleDeleteConversation}
							onStop={handleStopGeneration}
						/>
					</Sidebar.MenuItem>
				{/each}

				{#if filteredConversations.length === 0}
					<div class="px-2 py-4 text-center">
						<p class="mb-4 p-4 text-sm text-muted-foreground">
							{searchQuery.length > 0
								? 'No results found'
								: isSearchModeActive
									? 'Start typing to see results'
									: 'No conversations yet'}
						</p>
					</div>
				{/if}
			</Sidebar.Menu>
		</Sidebar.GroupContent>
	</Sidebar.Group>
</ScrollArea>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete Conversation"
	description={selectedConversation
		? `Are you sure you want to delete "${selectedConversation.name}"? This action cannot be undone and will permanently remove all messages in this conversation.`
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
/>

<AlertDialog.Root bind:open={showEditDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Edit Conversation Name</AlertDialog.Title>
			<AlertDialog.Description>
				<Input
					class="mt-4 text-foreground"
					onkeydown={(e) => {
						if (e.key === 'Enter') {
							e.preventDefault();
							handleConfirmEdit();
						}
					}}
					placeholder="Enter a new name"
					type="text"
					bind:value={editedName}
				/>
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel
				onclick={() => {
					showEditDialog = false;
					selectedConversation = null;
				}}>Cancel</AlertDialog.Cancel
			>
			<AlertDialog.Action onclick={handleConfirmEdit}>Save</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
