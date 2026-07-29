<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Lock } from '@lucide/svelte';
	import { encryptionStore } from '$lib/stores/encryption.svelte';

	let passphrase = $state('');
	let unlocking = $state(false);
	let failed = $state(false);

	const canSubmit = $derived(passphrase.length > 0 && !unlocking);

	async function handleSubmit(event: Event) {
		event.preventDefault();
		if (!canSubmit) return;

		unlocking = true;
		failed = false;
		try {
			const unlocked = await encryptionStore.unlockWithPassphrase(passphrase);
			failed = !unlocked;
			if (unlocked) {
				passphrase = '';
			}
		} finally {
			unlocking = false;
		}
	}
</script>

<AlertDialog.Root open={encryptionStore.needsUnlock}>
	<AlertDialog.Content
		onEscapeKeydown={(e) => e.preventDefault()}
		onInteractOutside={(e) => e.preventDefault()}
	>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Lock class="h-5 w-5" />
				Unlock your conversations
			</AlertDialog.Title>

			<AlertDialog.Description>
				Your conversations are encrypted. Enter your passphrase to unlock them for this session.
			</AlertDialog.Description>
		</AlertDialog.Header>

		<form onsubmit={handleSubmit} class="space-y-2 pt-2 pb-4">
			<label for="encryption-unlock-input" class="text-sm font-medium text-muted-foreground">
				Passphrase
			</label>

			<Input
				id="encryption-unlock-input"
				type="password"
				bind:value={passphrase}
				placeholder="Passphrase"
				autocomplete="current-password"
				aria-invalid={failed}
			/>

			{#if failed}
				<p class="text-sm text-destructive">Wrong passphrase. Please try again.</p>
			{/if}
		</form>

		<AlertDialog.Footer>
			<Button type="button" onclick={handleSubmit} disabled={!canSubmit}>
				{unlocking ? 'Unlocking...' : 'Unlock'}
			</Button>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
