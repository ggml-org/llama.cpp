<script lang="ts">
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmation } from '$lib/components/app';

	interface Props {
		open: boolean;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { onCancel, onConfirm, open = $bindable() }: Props = $props();
</script>

<DialogConfirmation
	bind:open
	cancelText="Cancel"
	confirmText="Delete"
	description="Are you sure you want to delete this conversation? This action cannot be undone and will permanently remove all messages in this conversation."
	icon={Trash2}
	{onCancel}
	{onConfirm}
	title="Delete Conversation"
	variant="destructive"
/>
