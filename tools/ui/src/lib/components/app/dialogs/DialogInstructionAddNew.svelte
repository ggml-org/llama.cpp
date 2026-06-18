<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { instructionsStore } from '$lib/stores/instructions.svelte';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
		initialContent?: string;
		onAddToLibraryComplete?: (instructionId: string, title: string) => void;
	}

	let { open = $bindable(), onOpenChange, initialContent = '', onAddToLibraryComplete }: Props = $props();

	let title = $state('');
	let content = $state('');
	let titleError = $derived.by(() => (!title.trim() ? 'Title is required' : null));

	$effect(() => {
		if (open && initialContent) {
			content = initialContent;
		}
	});

	function handleOpenChange(value: boolean) {
		if (!value) {
			title = '';
			content = '';
		}
		open = value;
		onOpenChange?.(value);
	}

	function saveNewInstruction() {
		if (titleError) return;

		const newInstruction = instructionsStore.addInstruction({ title: title.trim(), content: content.trim() });
		handleOpenChange(false);
		onAddToLibraryComplete?.(newInstruction.id, newInstruction.title);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-lg">
		<Dialog.Header>
			<Dialog.Title>Add New Instruction</Dialog.Title>
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
					placeholder="Enter instruction content..."
					bind:value={content}
				/>
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>
				Cancel
			</Button>

			<Button variant="default" size="sm" onclick={saveNewInstruction} disabled={!!titleError}>
				Add
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
