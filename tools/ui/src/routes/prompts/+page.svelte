<script lang="ts">
	import { PromptsCard } from '$lib/components/app';
	import { DialogPromptAddNew, DialogConfirmation } from '$lib/components/app/dialogs';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { Plus, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let isAdding = $state(false);
	let showDeleteDialog = $state(false);
	let deleteTarget = $state<{ id: string; title: string } | null>(null);

	let prompts = $derived(promptsStore.getPrompts());

	function handleDelete(id: string, title: string) {
		deleteTarget = { id, title };
		showDeleteDialog = true;
	}

	async function handleConfirmDelete() {
		if (deleteTarget) {
			await promptsStore.deletePrompt(deleteTarget.id);
			deleteTarget = null;
		}
		showDeleteDialog = false;
	}

	async function handleStartNewChat(id: string, title: string, content: string) {
		try {
			await conversationsStore.createConversation();
			await chatStore.addSystemPromptWithContent(content, id, title);
		} catch (error) {
			console.error('Failed to start new chat with prompt:', error);
		}
	}
</script>

<div class="mx-auto w-full p-4 md:p-8 md:py-8">
	<div class="mb-6 flex items-center gap-4">
		<h1 class="text-2xl font-semibold">Prompts</h1>

		{#if prompts.length > 0}
			<Button variant="outline" size="sm" onclick={() => (isAdding = true)}>
				<Plus class="h-4 w-4" />
				Add new prompt
			</Button>
		{/if}
	</div>

	{#if prompts.length === 0}
		<div class="flex flex-col items-center justify-center py-16">
			<div class="mb-4 rounded-full bg-muted p-4">
				<Plus class="h-6 w-6 text-muted-foreground" />
			</div>
			<h2 class="text-lg font-medium text-foreground">No prompts yet</h2>
			<p class="mt-1 text-sm text-muted-foreground">Create your first prompt to get started.</p>
			<Button class="mt-4" onclick={() => (isAdding = true)}>
				<Plus class="h-4 w-4" />
				Add new prompt
			</Button>
		</div>
	{:else}
		<div
			class="mt-4 grid gap-6"
			style="grid-template-columns: repeat(auto-fill, minmax(min(32rem, calc(100dvw - 2rem)), 1fr));"
		>
			{#each prompts as prompt (prompt.id)}
				<PromptsCard
					id={prompt.id}
					title={prompt.title}
					content={prompt.content}
					lastModified={prompt.lastModified}
					onDelete={() => handleDelete(prompt.id, prompt.title)}
					onStartNewChat={() => handleStartNewChat(prompt.id, prompt.title, prompt.content)}
				/>
			{/each}
		</div>
	{/if}
</div>

<DialogPromptAddNew bind:open={isAdding} />

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete prompt?"
	description="This will permanently remove the prompt. This action cannot be undone."
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => (showDeleteDialog = false)}
/>
