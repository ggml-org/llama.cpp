<script lang="ts">
	import { Eye, KeyRound, RefreshCw, ShieldCheck, TriangleAlert } from '@lucide/svelte';

	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Checkbox } from '$lib/components/ui/checkbox';

	interface Props {
		open: boolean;
		onConfirm: () => void;
		onCancel: () => void;
	}

	let { open = $bindable(), onConfirm, onCancel }: Props = $props();

	let acknowledged = $state(false);

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			acknowledged = false;
			onCancel();
		}
	}

	function handleConfirm() {
		acknowledged = false;
		onConfirm();
	}
</script>

<AlertDialog.Root {open} onOpenChange={handleOpenChange}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<ShieldCheck class="h-5 w-5" />
				Enable encryption?
			</AlertDialog.Title>

			<AlertDialog.Description>
				Your conversations will be encrypted at rest in this browser. Make sure you understand the
				consequences:
			</AlertDialog.Description>
		</AlertDialog.Header>

		<ul class="grid gap-2 text-sm">
			<li class="flex items-start gap-2">
				<KeyRound class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
				<span>Your passphrase is never stored. It cannot be recovered or reset.</span>
			</li>
			<li class="flex items-start gap-2">
				<TriangleAlert class="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
				<span>
					If you forget it, your conversations are permanently unreadable - by you and by anyone
					else.
				</span>
			</li>
			<li class="flex items-start gap-2">
				<RefreshCw class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
				<span>
					You will be asked for it again after 5 minutes of inactivity (configurable in Security
					settings once enabled).
				</span>
			</li>
			<li class="flex items-start gap-2">
				<Eye class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
				<span>Timestamps and message structure stay visible.</span>
			</li>
		</ul>

		<label class="flex items-start gap-2 text-sm font-medium">
			<Checkbox bind:checked={acknowledged} class="mt-0.5" />
			<span
				>I understand that forgetting the passphrase means permanently losing my conversations</span
			>
		</label>

		<AlertDialog.Footer>
			<AlertDialog.Cancel onclick={onCancel}>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action disabled={!acknowledged} onclick={handleConfirm}>
				Encrypt conversations
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
