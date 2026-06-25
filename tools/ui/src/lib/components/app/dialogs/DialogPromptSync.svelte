<script lang="ts">
	import { ArrowRight } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { CodeDiff } from '$lib/components/app/content';

	interface Props {
		open: boolean;
		promptTitle?: string;
		currentTitle?: string;
		currentContent: string;
		updatedContent?: string;
		updatedTitle?: string;
		editableTitle?: boolean;
		onOpenChange?: (open: boolean) => void;
		onUpdate?: (title?: string) => void;
	}

	let {
		open = $bindable(),
		promptTitle = 'this prompt',
		currentTitle,
		currentContent = '',
		updatedContent = '',
		updatedTitle = $bindable(),
		editableTitle = false,
		onOpenChange,
		onUpdate
	}: Props = $props();

	let libraryRenamed = $derived(
		!!currentTitle && !!promptTitle && promptTitle !== 'this prompt' && currentTitle !== promptTitle
	);

	$effect(() => {
		if (open && editableTitle && updatedTitle === undefined) {
			updatedTitle = promptTitle;
		}
	});

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}

	function handleUpdate() {
		onUpdate?.(editableTitle ? updatedTitle : undefined);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-3xl">
		<Dialog.Header>
			<Dialog.Title>
				{editableTitle
					? 'Update prompt in library and conversation?'
					: 'Update prompt in this conversation?'}
			</Dialog.Title>

			<Dialog.Description>
				{editableTitle
					? 'Review the changes below and update the library prompt title and content.'
					: `The library version of <strong>${promptTitle}</strong> has changed. Review the changes below, then update this conversation to match the library.`}
			</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 py-2">
			{#if libraryRenamed}
				<div
					class="flex items-center gap-2 rounded-md border border-dashed bg-muted/40 px-3 py-2 text-sm"
				>
					<span class="font-medium text-muted-foreground">Title renamed:</span>

					<span class="text-muted-foreground line-through">{currentTitle}</span>

					<ArrowRight class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

					<span class="font-medium">{promptTitle}</span>
				</div>
			{/if}

			{#if editableTitle}
				<div class="space-y-2">
					<Label for="sync-prompt-title" class="text-sm font-medium">Title</Label>

					<Input id="sync-prompt-title" type="text" bind:value={updatedTitle} class="w-full" />
				</div>
			{/if}

			{#if updatedContent && currentContent !== updatedContent}
				<div class="overflow-x-auto">
					<CodeDiff oldContent={currentContent} newContent={updatedContent} maxHeight="50vh" />
				</div>
			{/if}
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button variant="default" size="sm" onclick={handleUpdate}>Update</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
