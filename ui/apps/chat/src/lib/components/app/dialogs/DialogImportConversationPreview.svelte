<script lang="ts">
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import ChatMessages from '#lib/components/app/chat/ChatMessages/ChatMessages.svelte';

	interface Props {
		open: boolean;
		messages?: DatabaseMessage[];
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { messages = [], onCancel, onConfirm, open = $bindable() }: Props = $props();

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			onCancel();
		}
	}
</script>

<AlertDialog.Root onOpenChange={handleOpenChange} {open}>
	<AlertDialog.Content class="sm:max-w-3xl">
		<AlertDialog.Header>
			<AlertDialog.Title>Import conversation?</AlertDialog.Title>
		</AlertDialog.Header>

		<div
			class="max-h-[60vh] overflow-y-auto rounded-md border border-border/30 bg-muted/50 p-4 matte-well dark:border-border/20 dark:bg-muted/80"
		>
			<ChatMessages {messages} />
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Cancel onclick={onCancel}>Cancel</AlertDialog.Cancel>

			<AlertDialog.Action onclick={onConfirm}>Import</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
