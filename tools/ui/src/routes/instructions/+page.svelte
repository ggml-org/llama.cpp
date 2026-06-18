<script lang="ts">
	import { InstructionsCard } from '$lib/components/app';
	import { DialogInstructionAddNew, DialogConfirmation } from '$lib/components/app/dialogs';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { instructionsStore } from '$lib/stores/instructions.svelte';
	import { Plus, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	let isAdding = $state(false);
	let showDeleteDialog = $state(false);
	let deleteTarget = $state<{ id: string; title: string } | null>(null);
	let editTarget = $state<string | null>(null);

	let instructions = $derived(instructionsStore.getInstructions());

	function handleDelete(id: string, title: string) {
		deleteTarget = { id, title };
		showDeleteDialog = true;
	}

	function handleConfirmDelete() {
		if (deleteTarget) {
			instructionsStore.deleteInstruction(deleteTarget.id);
			deleteTarget = null;
		}
		showDeleteDialog = false;
	}

	function handleEdit(id: string) {
		editTarget = id;
	}

	async function handleStartNewChat(id: string, title: string, content: string) {
		try {
			await conversationsStore.createConversation();
			await chatStore.addSystemPromptWithContent(content, id, title);
		} catch (error) {
			console.error('Failed to start new chat with instruction:', error);
		}
	}
</script>

<div class="mx-auto w-full p-4 md:p-8 md:py-8">
	<div class="mb-6 flex items-center gap-4">
		<h1 class="text-2xl font-semibold">Instructions</h1>

		{#if instructions.length > 0}
			<Button variant="outline" size="sm" onclick={() => (isAdding = true)}>
				<Plus class="h-4 w-4" />
				Add new instruction
			</Button>
		{/if}
	</div>

	{#if instructions.length === 0}
		<div class="flex flex-col items-center justify-center py-16">
			<div class="mb-4 rounded-full bg-muted p-4">
				<Plus class="h-6 w-6 text-muted-foreground" />
			</div>
			<h2 class="text-lg font-medium text-foreground">No instructions yet</h2>
			<p class="mt-1 text-sm text-muted-foreground">
				Create your first instruction to get started.
			</p>
			<Button class="mt-4" onclick={() => (isAdding = true)}>
				<Plus class="h-4 w-4" />
				Add new instruction
			</Button>
		</div>
	{:else}
		<div
			class="mt-4 grid gap-6"
			style="grid-template-columns: repeat(auto-fill, minmax(min(32rem, calc(100dvw - 2rem)), 1fr));"
		>
			{#each instructions as instruction (instruction.id)}
				<InstructionsCard
					id={instruction.id}
					title={instruction.title}
					content={instruction.content}
					lastModified={instruction.lastModified}
					onDelete={() => handleDelete(instruction.id, instruction.title)}
					onEdit={handleEdit}
					onStartNewChat={() => handleStartNewChat(instruction.id, instruction.title, instruction.content)}
				/>
			{/each}
		</div>
	{/if}
</div>

<DialogInstructionAddNew bind:open={isAdding} />

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete instruction?"
	description="This will permanently remove the instruction. This action cannot be undone."
	confirmText="Delete"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => (showDeleteDialog = false)}
/>
