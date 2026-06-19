<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { promptsStore } from '$lib/stores/prompts.svelte';
	import { CategoryCombobox } from '$lib/components/app';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
		initialContent?: string;
		onAddToLibraryComplete?: (promptId: string, title: string) => void;
	}

	let {
		open = $bindable(),
		onOpenChange,
		initialContent = '',
		onAddToLibraryComplete
	}: Props = $props();

	let title = $state('');
	let content = $state('');
	let category = $state('');
	let titleError = $derived.by(() => (!title.trim() ? 'Title is required' : null));
	let existingCategories = $derived(promptsStore.getCategories());

	$effect(() => {
		if (open && initialContent) {
			content = initialContent;
		}
	});

	function handleOpenChange(value: boolean) {
		if (!value) {
			title = '';
			content = '';
			category = '';
		}
		open = value;
		onOpenChange?.(value);
	}

	async function saveNewPrompt() {
		if (titleError) return;

		const trimmedCategory = category.trim();
		const newPrompt = await promptsStore.addPrompt({
			title: title.trim(),
			content: content.trim(),
			...(trimmedCategory ? { category: trimmedCategory } : {})
		});
		handleOpenChange(false);
		onAddToLibraryComplete?.(newPrompt.id, newPrompt.title);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Add New Prompt</Dialog.Title>
		</Dialog.Header>

		<div class="space-y-4 py-4">
			<div class="space-y-2">
				<label class="text-sm font-medium text-foreground">Title</label>
				<input
					class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
					placeholder="e.g. Code Review Assistant"
					type="text"
					bind:value={title}
				/>
				{#if titleError}
					<p class="text-xs text-destructive">{titleError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<label class="text-sm font-medium text-foreground">Content</label>
				<textarea
					class="flex min-h-[120px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
					placeholder="Enter prompt content..."
					bind:value={content}
				></textarea>
			</div>

			<div class="space-y-2">
				<label class="text-sm font-medium text-foreground">Category</label>
				<CategoryCombobox bind:value={category} categories={existingCategories} />
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button variant="default" size="sm" onclick={saveNewPrompt} disabled={!!titleError}>
				Add
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
