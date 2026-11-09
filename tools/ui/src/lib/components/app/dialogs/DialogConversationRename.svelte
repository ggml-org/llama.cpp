<script lang="ts">
	import { Pencil } from '@lucide/svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';

	interface Props {
		open: boolean;
		currentTitle: string;
		value: string;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let {
		currentTitle,
		onCancel,
		onConfirm,
		open = $bindable(),
		value = $bindable('')
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

		<form class="space-y-2 pt-2 pb-4" onsubmit={handleSubmit}>
			<label class="text-sm font-medium text-muted-foreground" for="conversation-rename-input">
				Conversation title
			</label>

			<Input
				bind:ref={inputRef}
				bind:value
				autocomplete="off"
				autocorrect="off"
				id="conversation-rename-input"
				maxlength={200}
				placeholder="Conversation title"
				spellcheck={false}
			/>
		</form>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

			<Button disabled={!canSubmit} onclick={handleSubmit} type="button">Save</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
