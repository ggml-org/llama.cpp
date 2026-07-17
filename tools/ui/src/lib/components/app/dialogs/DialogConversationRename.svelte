<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Pencil } from '@lucide/svelte';

	interface Props {
		open: boolean;
		currentTitle: string;
		value: string;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let {
		open = $bindable(),
		currentTitle,
		value = $bindable(''),
		onConfirm,
		onCancel
	}: Props = $props();

	let inputRef = $state<HTMLInputElement | null>(null);

	const canSubmit = $derived(value.trim().length > 0 && value.trim() !== currentTitle.trim());

	$effect(() => {
		if (open) {
			value = currentTitle;
			queueMicrotask(() => {
				inputRef?.focus();
				inputRef?.select();
			});
		}
	});

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			onCancel();
		}
	}

	function handleSubmit(event: Event) {
		event.preventDefault();
		if (!canSubmit) return;
		value = value.trim();
		onConfirm();
	}
</script>

<AlertDialog.Root bind:open onOpenChange={handleOpenChange}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Pencil class="h-5 w-5" />
				Rename conversation
			</AlertDialog.Title>

			<AlertDialog.Description>Choose a new title for this conversation.</AlertDialog.Description>
		</AlertDialog.Header>

		<form onsubmit={handleSubmit} class="space-y-2 pt-2 pb-4">
			<label for="conversation-rename-input" class="text-sm font-medium text-muted-foreground">
				Conversation title
			</label>

			<Input
				id="conversation-rename-input"
				bind:ref={inputRef}
				bind:value
				placeholder="Conversation title"
				maxlength={200}
				autocomplete="off"
				autocorrect="off"
				spellcheck={false}
			/>
		</form>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

			<Button type="button" onclick={handleSubmit} disabled={!canSubmit}>Save</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
