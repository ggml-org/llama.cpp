<script lang="ts">
	import { DialogConversationRename, SidebarNavigationConversationItem } from '$lib/components/app';
	import { chatStore, conversationsStore } from '$lib/stores';
	import type { DatabaseConversation } from '$lib/types';
	import { formatRelativeTime } from '$lib/utils';

	interface Props {
		conversations: DatabaseConversation[];
		onOpen: (id: string) => void;
	}

	let { conversations, onOpen }: Props = $props();

	// the rows are the sidebar ones, so the actions they expose need the same
	// rename and delete wiring the sidebar list owns
	let renameDialogOpen = $state(false);
	let renameDraft = $state('');
	let renameOriginalTitle = $state('');
	let renameTargetId = $state<string | null>(null);

	function openRename(id: string) {
		const conversation = conversations.find((item) => item.id === id);

		if (!conversation) return;

		renameTargetId = id;
		renameOriginalTitle = conversation.name;
		renameDraft = conversation.name;
		renameDialogOpen = true;
	}

	async function confirmRename() {
		const id = renameTargetId;

		if (!id) return;

		const nextName = renameDraft.trim();

		if (!nextName || nextName === renameOriginalTitle.trim()) return;

		await conversationsStore.updateConversationName(id, nextName);

		renameDialogOpen = false;
		renameTargetId = null;
	}

	function cancelRename() {
		renameDialogOpen = false;
		renameTargetId = null;
		renameDraft = '';
		renameOriginalTitle = '';
	}

	async function deleteConversation(id: string) {
		const conversation = conversations.find((item) => item.id === id);

		if (!conversation) return;

		const confirmed = window.confirm(
			`Delete "${conversation.name}"? This action cannot be undone.`
		);

		if (!confirmed) return;

		await conversationsStore.deleteConversation(id, { deleteWithForks: false });
	}
</script>

<div class="flex flex-col">
	<h2 class="px-1 pb-1 text-sm font-medium text-muted-foreground">Recents</h2>

	<ul class="flex min-w-0 flex-col">
		{#each conversations as conversation (conversation.id)}
			<li class="min-w-0">
				<SidebarNavigationConversationItem
					{conversation}
					meta={formatRelativeTime(conversation.lastModified)}
					onDelete={deleteConversation}
					onEdit={openRename}
					onSelect={onOpen}
					onStop={(id) => chatStore.stopGenerationForChat(id)}
					size="lg"
				/>
			</li>
		{/each}
	</ul>
</div>

<DialogConversationRename
	bind:open={renameDialogOpen}
	bind:value={renameDraft}
	currentTitle={renameOriginalTitle}
	onCancel={cancelRename}
	onConfirm={confirmRename}
/>
