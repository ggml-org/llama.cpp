<script lang="ts">
	import { onMount } from 'svelte';
	import { Lock, LockOpen, ShieldCheck, ShieldOff } from '@lucide/svelte';
	import { toast } from 'svelte-sonner';

	import { DialogConfirmation } from '$lib/components/app';
	import SettingsGroup from '$lib/components/app/settings/SettingsGroup.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { DatabaseService } from '$lib/services/database.service';
	import { encryptionStore } from '$lib/stores/encryption.svelte';

	let newPassphrase = $state('');
	let newPassphraseConfirm = $state('');
	let enabling = $state(false);

	let currentPassphrase = $state('');
	let nextPassphrase = $state('');
	let nextPassphraseConfirm = $state('');
	let changing = $state(false);

	let showDisableDialog = $state(false);
	let disabling = $state(false);

	const canEnable = $derived(
		newPassphrase.length > 0 && newPassphrase === newPassphraseConfirm && !enabling
	);
	const canChange = $derived(
		currentPassphrase.length > 0 &&
			nextPassphrase.length > 0 &&
			nextPassphrase === nextPassphraseConfirm &&
			!changing
	);

	onMount(() => encryptionStore.refresh());

	async function handleEnable() {
		if (!canEnable) return;

		enabling = true;
		try {
			await encryptionStore.setupWithPassphrase(newPassphrase);
			await DatabaseService.encryptAllStoredData();
			newPassphrase = '';
			newPassphraseConfirm = '';
			toast.success('Encryption enabled');
		} catch (error) {
			console.error('Failed to enable encryption:', error);
			toast.error('Failed to enable encryption');
		} finally {
			enabling = false;
		}
	}

	async function handleChangePassphrase() {
		if (!canChange) return;

		changing = true;
		try {
			const changed = await encryptionStore.changePassphrase(currentPassphrase, nextPassphrase);
			if (changed) {
				currentPassphrase = '';
				nextPassphrase = '';
				nextPassphraseConfirm = '';
				toast.success('Passphrase changed');
			} else {
				toast.error('Current passphrase is wrong');
			}
		} finally {
			changing = false;
		}
	}

	async function handleDisableConfirm() {
		showDisableDialog = false;

		disabling = true;
		try {
			await DatabaseService.decryptAllStoredData();
			encryptionStore.disable();
			toast.success('Encryption disabled');
		} catch (error) {
			console.error('Failed to disable encryption:', error);
			toast.error('Failed to disable encryption');
		} finally {
			disabling = false;
		}
	}
</script>

{#if !encryptionStore.isSupported}
	<p class="text-sm text-muted-foreground">
		Encryption is not available: the Web Crypto API requires a secure context (HTTPS or localhost).
	</p>
{:else if !encryptionStore.isEnabled}
	<SettingsGroup title="Encrypt conversations">
		<div class="grid gap-1">
			<p class="mb-4 text-sm text-muted-foreground">
				Encrypt your conversations at rest in this browser with a passphrase. Conversation names,
				message content and attachments are encrypted; timestamps and the conversation structure
				stay readable. The passphrase is held in memory only and is required on every visit - if you
				forget it, your conversations cannot be recovered.
			</p>

			<form
				onsubmit={(e) => {
					e.preventDefault();
					handleEnable();
				}}
				class="grid max-w-sm gap-2"
			>
				<Input
					type="password"
					bind:value={newPassphrase}
					placeholder="Passphrase"
					autocomplete="new-password"
					aria-label="Passphrase"
				/>
				<Input
					type="password"
					bind:value={newPassphraseConfirm}
					placeholder="Confirm passphrase"
					autocomplete="new-password"
					aria-label="Confirm passphrase"
				/>

				{#if newPassphraseConfirm.length > 0 && newPassphrase !== newPassphraseConfirm}
					<p class="text-sm text-destructive">Passphrases do not match.</p>
				{/if}

				<Button type="submit" disabled={!canEnable} class="justify-self-start">
					<ShieldCheck class="mr-2 h-4 w-4" />
					{enabling ? 'Encrypting...' : 'Enable encryption'}
				</Button>
			</form>
		</div>
	</SettingsGroup>
{:else}
	<div class="space-y-10">
		<SettingsGroup title="Session">
			<div class="grid gap-1">
				<p class="mb-4 text-sm text-muted-foreground">
					Encryption is on and this session is {encryptionStore.isUnlocked ? 'unlocked' : 'locked'}.
					Locking drops the key from memory; the passphrase is required again to read or write
					conversations.
				</p>

				{#if encryptionStore.isUnlocked}
					<Button
						variant="outline"
						class="justify-self-start"
						onclick={() => encryptionStore.lock()}
					>
						<Lock class="mr-2 h-4 w-4" />
						Lock now
					</Button>
				{:else}
					<p class="text-sm text-muted-foreground">
						<LockOpen class="mr-1 inline h-4 w-4" />
						Unlock with your passphrase to continue.
					</p>
				{/if}
			</div>
		</SettingsGroup>

		<SettingsGroup title="Change passphrase">
			<form
				onsubmit={(e) => {
					e.preventDefault();
					handleChangePassphrase();
				}}
				class="grid max-w-sm gap-2"
			>
				<Input
					type="password"
					bind:value={currentPassphrase}
					placeholder="Current passphrase"
					autocomplete="current-password"
					aria-label="Current passphrase"
				/>
				<Input
					type="password"
					bind:value={nextPassphrase}
					placeholder="New passphrase"
					autocomplete="new-password"
					aria-label="New passphrase"
				/>
				<Input
					type="password"
					bind:value={nextPassphraseConfirm}
					placeholder="Confirm new passphrase"
					autocomplete="new-password"
					aria-label="Confirm new passphrase"
				/>

				{#if nextPassphraseConfirm.length > 0 && nextPassphrase !== nextPassphraseConfirm}
					<p class="text-sm text-destructive">Passphrases do not match.</p>
				{/if}

				<Button type="submit" variant="outline" disabled={!canChange} class="justify-self-start">
					Change passphrase
				</Button>
			</form>
		</SettingsGroup>

		<SettingsGroup title="Disable encryption">
			<div class="grid gap-1">
				<p class="mb-4 text-sm text-muted-foreground">
					Decrypt all conversations and store them as plaintext again.
				</p>

				<Button
					variant="destructive"
					class="justify-self-start"
					disabled={disabling || !encryptionStore.isUnlocked}
					onclick={() => (showDisableDialog = true)}
				>
					<ShieldOff class="mr-2 h-4 w-4" />
					{disabling ? 'Decrypting...' : 'Disable encryption'}
				</Button>
			</div>
		</SettingsGroup>
	</div>
{/if}

<DialogConfirmation
	bind:open={showDisableDialog}
	title="Disable encryption"
	description="All conversations will be decrypted and stored as plaintext. Continue?"
	confirmText="Disable"
	variant="destructive"
	icon={ShieldOff}
	onConfirm={handleDisableConfirm}
	onCancel={() => (showDisableDialog = false)}
/>
