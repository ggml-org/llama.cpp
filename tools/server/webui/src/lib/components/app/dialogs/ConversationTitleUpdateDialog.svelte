<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		open: boolean;
		currentTitle: string;
		newTitle: string;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { open = $bindable(), currentTitle, newTitle, onConfirm, onCancel }: Props = $props();
</script>

<AlertDialog.Root bind:open>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Update Conversation Title?</AlertDialog.Title>

			<AlertDialog.Description>
				Do you want to update the conversation title to match the first message content?
			</AlertDialog.Description>
		</AlertDialog.Header>

		<div class="space-y-4 pt-2 pb-6">
			<div class="space-y-2">
				<p class="text-sm font-medium text-muted-foreground">Current title:</p>

				<p class="rounded-md bg-muted/50 p-3 text-sm font-medium">{currentTitle}</p>
			</div>

			<div class="space-y-2">
				<p class="text-sm font-medium text-muted-foreground">New title would be:</p>

				<p class="rounded-md bg-muted/50 p-3 text-sm font-medium">{newTitle}</p>
			</div>
		</div>

		<AlertDialog.Footer>
			<Button variant="outline" onclick={onCancel}>Keep Current Title</Button>

			<Button onclick={onConfirm}>Update Title</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
