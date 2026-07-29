<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Lock } from '@lucide/svelte';

	interface Props {
		open?: boolean;
		onConfirm: (unencrypted: boolean) => void;
		onCancel: () => void;
	}

	let { open = $bindable(false), onConfirm, onCancel }: Props = $props();

	let unencrypted = $state(false);

	let previousOpen = $state(false);

	$effect(() => {
		if (open && !previousOpen) {
			unencrypted = false;
		}

		previousOpen = open;
	});
</script>

<AlertDialog.Root bind:open>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Lock class="h-5 w-5" />
				Export conversations
			</AlertDialog.Title>

			<AlertDialog.Description>
				Encryption is enabled, so the export will be protected with your passphrase. You will need
				it to import the file, including on other devices.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<label class="flex items-center gap-2 pt-2 pb-4 text-sm">
			<Checkbox bind:checked={unencrypted} />
			Export unencrypted conversations
		</label>

		<AlertDialog.Footer>
			<AlertDialog.Cancel onclick={onCancel}>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action onclick={() => onConfirm(unencrypted)}>Export</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
