<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Eye,
		Lock,
		LockOpen,
		ShieldCheck,
		ShieldOff,
		Timer,
		TriangleAlert
	} from '@lucide/svelte';
	import { toast } from 'svelte-sonner';

	import * as Alert from '$lib/components/ui/alert';
	import * as Select from '$lib/components/ui/select';
	import { DialogConfirmation, DialogEncryptionEnable } from '$lib/components/app';
	import SettingsGroup from '$lib/components/app/settings/SettingsGroup.svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { DatabaseService } from '$lib/services/database.service';
	import { McpSecretsService } from '$lib/services/mcp-secrets.service';
	import { encryptionStore, IDLE_TIMEOUT_OPTIONS } from '$lib/stores/encryption.svelte';
	import type { IdleTimeoutMinutes } from '$lib/stores/encryption.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';

	let newPassphrase = $state('');
	let newPassphraseConfirm = $state('');
	let enabling = $state(false);

	let currentPassphrase = $state('');
	let nextPassphrase = $state('');
	let nextPassphraseConfirm = $state('');
	let changing = $state(false);

	let showEnableDialog = $state(false);
	let showDisableDialog = $state(false);
	let disabling = $state(false);

	const MIN_PASSPHRASE_LENGTH = 8;

	const canEnable = $derived(
		newPassphrase.length >= MIN_PASSPHRASE_LENGTH &&
			newPassphrase === newPassphraseConfirm &&
			!enabling
	);
	const canChange = $derived(
		currentPassphrase.length > 0 &&
			nextPassphrase.length >= MIN_PASSPHRASE_LENGTH &&
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
			await mcpStore.loadSecrets();
			await McpSecretsService.persist();
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
			await McpSecretsService.persist({ plaintext: true });
			encryptionStore.disable();
			toast.success('Encryption disabled');
		} catch (error) {
			console.error('Failed to disable encryption:', error);
			toast.error('Failed to disable encryption');
		} finally {
			disabling = false;
		}
	}

	function handleIdleTimeoutChange(value: string): void {
		const minutes = Number(value) as IdleTimeoutMinutes;
		encryptionStore.setIdleTimeout(minutes);
	}
</script>

{#if !encryptionStore.isSupported}
	<p class="text-sm text-muted-foreground">
		Encryption is not available: the Web Crypto API requires a secure context (HTTPS or localhost).
	</p>
{:else if !encryptionStore.isEnabled}
	<SettingsGroup title="Encrypt conversations">
		<div class="grid gap-4">
			<p class="text-sm text-muted-foreground">
				Encrypt your conversations at rest in this browser with a passphrase.
			</p>

			<div class="grid gap-4 text-sm sm:grid-cols-2">
				<div>
					<p class="mb-1 flex items-center gap-1.5 font-medium">
						<Lock class="h-4 w-4" />
						Encrypted
					</p>
					<ul class="list-disc pl-5 text-muted-foreground">
						<li>Conversation titles</li>
						<li>Message content</li>
						<li>Attachments</li>
					</ul>
				</div>

				<div>
					<p class="mb-1 flex items-center gap-1.5 font-medium">
						<Eye class="h-4 w-4" />
						Stays visible
					</p>
					<ul class="list-disc pl-5 text-muted-foreground">
						<li>Timestamps</li>
						<li>Message counts and structure</li>
					</ul>
				</div>
			</div>

			<Alert.Root
				class="max-w-xl border-amber-500/50 bg-amber-500/10 text-amber-700 dark:text-amber-400"
			>
				<TriangleAlert />
				<Alert.Title>No recovery if you forget the passphrase</Alert.Title>
				<Alert.Description class="text-amber-700/90 dark:text-amber-400/90">
					It is never stored and cannot be reset. Without it, your conversations cannot be decrypted
					- by you or by anyone else. You will need it again after inactivity (5 minutes by default,
					configurable below).
				</Alert.Description>
			</Alert.Root>

			<form
				onsubmit={(e) => {
					e.preventDefault();
					if (canEnable) showEnableDialog = true;
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

				{#if newPassphrase.length > 0 && newPassphrase.length < MIN_PASSPHRASE_LENGTH}
					<p class="text-sm text-destructive">Use at least {MIN_PASSPHRASE_LENGTH} characters.</p>
				{:else if newPassphraseConfirm.length > 0 && newPassphrase !== newPassphraseConfirm}
					<p class="text-sm text-destructive">Passphrases do not match.</p>
				{/if}

				<Button type="submit" disabled={!canEnable} class="justify-self-start">
					<ShieldCheck class="mr-2 h-4 w-4" />
					{enabling ? 'Encrypting...' : 'Enable encryption'}
				</Button>
			</form>
		</div>
	</SettingsGroup>

	<DialogEncryptionEnable
		bind:open={showEnableDialog}
		onConfirm={handleEnable}
		onCancel={() => (showEnableDialog = false)}
	/>
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

		<SettingsGroup title="Auto-lock">
			<div class="grid gap-1">
				<p class="mb-4 max-w-xl text-sm text-muted-foreground">
					Skip the passphrase prompt while you stay within this window, even across page refreshes.
					The key is kept in this browser's storage during the window and removed when it expires -
					until then, anyone using this browser can read your conversations without the passphrase.
					Never keeps you unlocked until you lock manually.
				</p>

				<div class="flex items-center gap-3 max-w-xs">
					<Timer class="h-4 w-4 text-muted-foreground shrink-0" />
					<Select.Root
						type="single"
						value={String(encryptionStore.idleTimeoutMinutes)}
						onValueChange={handleIdleTimeoutChange}
					>
						<Select.Trigger class="w-[180px]">
							{IDLE_TIMEOUT_OPTIONS.find((o) => o.value === encryptionStore.idleTimeoutMinutes)
								?.label ?? '5 minutes'}
						</Select.Trigger>
						<Select.Content>
							{#each IDLE_TIMEOUT_OPTIONS as option (option.value)}
								<Select.Item value={String(option.value)}>
									{option.label}
								</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>
			</div>
		</SettingsGroup>

		<SettingsGroup title="Change passphrase">
			<p class="mb-4 max-w-xl text-sm text-muted-foreground">
				Conversations stay encrypted - only the passphrase changes. If you forget the new
				passphrase, they cannot be recovered.
			</p>

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

				{#if nextPassphrase.length > 0 && nextPassphrase.length < MIN_PASSPHRASE_LENGTH}
					<p class="text-sm text-destructive">Use at least {MIN_PASSPHRASE_LENGTH} characters.</p>
				{:else if nextPassphraseConfirm.length > 0 && nextPassphrase !== nextPassphraseConfirm}
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
	description="All conversations will be decrypted and stored as plaintext in this browser. Anyone using this browser will be able to read them. Continue?"
	confirmText="Disable"
	variant="destructive"
	icon={ShieldOff}
	onConfirm={handleDisableConfirm}
	onCancel={() => (showDisableDialog = false)}
/>
