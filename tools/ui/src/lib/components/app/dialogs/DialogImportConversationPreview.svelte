<script lang="ts">
	import { MessageSquarePlus } from '@lucide/svelte';
	import ChatMessages from '$lib/components/app/chat/ChatMessages/ChatMessages.svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';

	interface Props {
		open: boolean;
		conversationName?: string;
		messages?: DatabaseMessage[];
		onConfirm: () => void;
		onCancel: () => void;
	}

	let {
		conversationName = '',
		messages = [],
		onCancel,
		onConfirm,
		open = $bindable()
	}: Props = $props();

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			onCancel();
		}
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content class="sm:max-w-3xl">
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<MessageSquarePlus class="h-5 w-5" />

				Import conversation?
			</AlertDialog.Title>

			<AlertDialog.Description>
				Preview of
				<span class="font-medium">"{conversationName || 'Untitled conversation'}"</span>. Confirm to
				import it into your library.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="max-h-[60vh] overflow-y-auto rounded-md border">
			<ChatMessages {messages} />
		</div>

		<AlertDialog.Footer>
			<AlertDialog.Cancel onclick={onCancel}>Cancel</AlertDialog.Cancel>

			<AlertDialog.Action onclick={onConfirm}>Import</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
