<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Textarea } from '$lib/components/ui/textarea';
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
	let contentError = $derived.by(() => (!content.trim() ? 'Content is required' : null));
	let saveError = $derived.by(() => titleError || contentError);
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
		if (saveError) return;

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
	<Dialog.Content class="sm:max-w-2xl">
		<Dialog.Header>
			<Dialog.Title>Add Prompt to library</Dialog.Title>

			<Dialog.Description>
				Save this system message to your library so you can reuse it across conversations.
			</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-6 py-2">
			<div class="space-y-2">
				<Label for="prompt-title" class="text-sm font-medium">Title</Label>

				<Input
					id="prompt-title"
					type="text"
					bind:value={title}
					placeholder="e.g. Code Review Assistant"
					class="w-full"
				/>

				{#if titleError}
					<p class="text-xs text-destructive">{titleError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<Label for="prompt-content" class="text-sm font-medium">Content</Label>

				<Textarea
					id="prompt-content"
					bind:value={content}
					placeholder="Enter prompt content..."
					class="min-h-[10rem] w-full"
				/>

				{#if contentError}
					<p class="text-xs text-destructive">{contentError}</p>
				{/if}
			</div>

			<div class="flex flex-col gap-2">
				<label for="prompt-category" class="text-sm font-medium">Category</label>

				<CategoryCombobox
					id="prompt-category"
					bind:value={category}
					categories={existingCategories}
				/>
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="outline" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button onclick={saveNewPrompt} disabled={!!saveError}>Add</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
