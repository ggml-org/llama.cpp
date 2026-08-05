<script lang="ts">
	import { Lock } from '@lucide/svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Input } from '$lib/components/ui/input';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		open?: boolean;
		/** Try to decrypt with the passphrase; resolves to false when it is wrong */
		onSubmit: (passphrase: string) => Promise<boolean>;
		onCancel: () => void;
	}

	let { open = $bindable(false), onSubmit, onCancel }: Props = $props();

	let passphrase = $state('');
	let failed = $state(false);
	let decrypting = $state(false);

	let previousOpen = $state(false);

	// Clear any stale passphrase/error each time the dialog is opened
	$effect(() => {
		if (open && !previousOpen) {
			passphrase = '';
			failed = false;
		}
		previousOpen = open;
	});

	async function handleSubmit() {
		if (!passphrase || decrypting) return;

		decrypting = true;
		failed = false;
		try {
			const ok = await onSubmit(passphrase);
			if (!ok) {
				failed = true;
				return;
			}
			passphrase = '';
		} finally {
			decrypting = false;
		}
	}
</script>

<AlertDialog.Root
	{open}
	onOpenChange={(isOpen) => {
		if (!isOpen) onCancel();
	}}
>
	<AlertDialog.Content
		onEscapeKeydown={(e) => e.preventDefault()}
		onInteractOutside={(e) => e.preventDefault()}
	>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Lock class="h-5 w-5" />
				Encrypted import
			</AlertDialog.Title>

			<AlertDialog.Description>
				This export is encrypted. Enter the passphrase that was used to protect it.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<form
			onsubmit={(e) => {
				e.preventDefault();
				void handleSubmit();
			}}
			class="space-y-2 pt-2 pb-4"
		>
			<Input
				type="password"
				bind:value={passphrase}
				placeholder="Passphrase"
				autocomplete="off"
				aria-label="Export passphrase"
				aria-invalid={failed}
			/>

			{#if failed}
				<p class="text-sm text-destructive">Wrong passphrase. Please try again.</p>
			{/if}
		</form>

		<AlertDialog.Footer>
			<Button variant="outline" type="button" onclick={onCancel}>Cancel</Button>

			<Button
				type="button"
				disabled={!passphrase || decrypting}
				onclick={() => void handleSubmit()}
			>
				{decrypting ? 'Decrypting...' : 'Decrypt'}
			</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
