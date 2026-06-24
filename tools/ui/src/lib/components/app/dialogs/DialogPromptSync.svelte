<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { CodeDiff } from '$lib/components/app/content';

	interface Props {
		open: boolean;
		promptTitle?: string;
		currentContent: string;
		updatedContent: string;
		onOpenChange?: (open: boolean) => void;
		onUpdate: () => void;
	}

	let {
		open = $bindable(),
		promptTitle = 'this prompt',
		currentContent = '',
		updatedContent = '',
		onOpenChange,
		onUpdate
	}: Props = $props();

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-3xl">
		<Dialog.Header>
			<Dialog.Title>Update prompt in this conversation?</Dialog.Title>

			<Dialog.Description>
				The library version of <strong>{promptTitle}</strong> has changed. Review the changes below, then
				update this conversation to match the library.
			</Dialog.Description>
		</Dialog.Header>

		<div class="overflow-x-auto py-4">
			<CodeDiff oldContent={currentContent} newContent={updatedContent} maxHeight="50vh" />
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button variant="default" size="sm" onclick={onUpdate}>Update</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
