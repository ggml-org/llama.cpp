<script lang="ts">
	import { Plus, Trash2 } from '@lucide/svelte';

	import { Button } from '$lib/components/ui/button';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import { ManageLayout, PromptsCard } from '$lib/components/app';
	import { SIDEBAR_ACTIONS_ITEMS } from '$lib/constants/ui';
	import { DialogPromptAddNew, DialogConfirmation } from '$lib/components/app/dialogs';

	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';

	const ALL = 'all';
	const OTHER = '__other__';

	let isAdding = $state(false);
	let showDeleteDialog = $state(false);
	let deleteTarget = $state<{ id: string; title: string } | null>(null);
	let selectedCategory = $state<string>(ALL);

	let categories = $derived(promptsStore.getCategories());
	let hasUncategorized = $derived(promptsStore.hasUncategorized());
	let showOther = $derived(categories.length > 0 && hasUncategorized);
	let prompts = $derived.by(() => {
		if (!selectedCategory || selectedCategory === ALL) return promptsStore.getPrompts();
		if (selectedCategory === OTHER) return promptsStore.getUncategorizedPrompts();
		return promptsStore.getPrompts(selectedCategory);
	});
	let hasAnyPrompts = $derived(promptsStore.getPrompts().length > 0);

	// Drop the selection if its category no longer exists
	$effect(() => {
		if (selectedCategory === OTHER && !hasUncategorized) {
			selectedCategory = ALL;
			return;
		}
		if (
			selectedCategory !== ALL &&
			selectedCategory !== OTHER &&
			!categories.includes(selectedCategory)
		) {
			selectedCategory = ALL;
		}
	});

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

<ManageLayout title="Prompts">
	{#snippet icon()}
		{@const Icon = SIDEBAR_ACTIONS_ITEMS.find((i) => i.tooltip === 'Prompts')?.icon}
		{#if Icon}
			<Icon class="h-5 w-5 md:h-6 md:w-6" />
		{/if}
	{/snippet}

	{#snippet actions()}
		{#if hasAnyPrompts}
			<Button
				variant="outline"
				size={isMobile.current ? 'lg' : 'default'}
				onclick={() => (isAdding = true)}
			>
				<Plus class="h-4 w-4" />
				Add new prompt
			</Button>
		{/if}
	{/snippet}

	{#if !hasAnyPrompts}
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
	{:else if prompts.length === 0}
		<div class="flex flex-col items-center justify-center py-16">
			<div class="mb-4 rounded-full bg-muted p-4">
				<Plus class="h-6 w-6 text-muted-foreground" />
			</div>
			<h2 class="text-lg font-medium text-foreground">No prompts in this category</h2>
			<p class="mt-1 text-sm text-muted-foreground">
				Pick a different category or add a new prompt here.
			</p>
			<Button class="mt-4" onclick={() => (isAdding = true)}>
				<Plus class="h-4 w-4" />
				Add new prompt
			</Button>
		</div>
	{:else}
		{#if categories.length > 0}
			<ToggleGroup.Root
				bind:value={selectedCategory}
				type="single"
				variant="outline"
				size="sm"
				spacing={1}
				class="mb-6 w-full flex-wrap"
			>
				<ToggleGroup.Item value={ALL} aria-label="Show all prompts">All</ToggleGroup.Item>
				{#each categories as category (category)}
					<ToggleGroup.Item value={category} aria-label={`Filter by ${category}`}>
						{category}
					</ToggleGroup.Item>
				{/each}
				{#if showOther}
					<ToggleGroup.Item value={OTHER} aria-label="Show uncategorized prompts">
						Other
					</ToggleGroup.Item>
				{/if}
			</ToggleGroup.Root>
		{/if}

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
					category={prompt.category}
					onDelete={() => handleDelete(prompt.id, prompt.title)}
					onStartNewChat={() => handleStartNewChat(prompt.id, prompt.title, prompt.content)}
				/>
			{/each}
		</div>
	{/if}

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
</ManageLayout>
