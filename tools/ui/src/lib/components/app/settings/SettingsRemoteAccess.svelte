<script lang="ts">
	import { fade } from 'svelte/transition';
	import { Wifi, WifiOff, Copy, Check, Users, AlertCircle, Loader2, RefreshCw } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Badge } from '$lib/components/ui/badge';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { SettingsGroup } from '$lib/components/app/settings';
	import { webrtcStore } from '$lib/stores/webrtc.svelte';

	// -- host state
	let codeCopied = $state(false);
	let showRegenerateDialog = $state(false);
	let regenerating = $state(false);

	// -- client state
	let joinInput = $state('');
	let joinError = $state('');
	let joining = $state(false);

	async function handleStartHost() {
		await webrtcStore.startHost();
	}

	function handleStopHost() {
		webrtcStore.stopHost();
	}

	function copyCode() {
		navigator.clipboard.writeText(webrtcStore.shareCode).then(() => {
			codeCopied = true;
			setTimeout(() => (codeCopied = false), 2000);
		});
	}

	async function handleRegenerateConfirm() {
		showRegenerateDialog = false;
		regenerating = true;
		try {
			await webrtcStore.regenerateCodes();
		} finally {
			regenerating = false;
		}
	}

	async function handleJoin() {
		joinError = '';
		const code = joinInput.trim().replace(/\s/g, '');
		if (code.length < 40) {
			joinError = 'Code must be 40 characters';
			return;
		}
		joining = true;
		try {
			await webrtcStore.joinAsClient(code);
		} catch (e) {
			joinError = e instanceof Error ? e.message : String(e);
		} finally {
			joining = false;
		}
	}

	function handleLeave() {
		webrtcStore.leaveAsClient();
		joinInput = '';
		joinError = '';
	}

	// Display the share code broken into 8-char blocks for readability
	function formatCode(code: string): string {
		return code.match(/.{1,8}/g)?.join(' ') ?? code;
	}
</script>

<div class="space-y-12" in:fade={{ duration: 150 }}>
	<!-- ------------------------------------------------------------------ -->
	<!-- HOST -->
	<!-- ------------------------------------------------------------------ -->
	<SettingsGroup title="Host">
		<div class="space-y-4">
			<p class="text-sm text-muted-foreground">
				Generate a code and share it so remote devices can connect to this instance.
			</p>

			<!-- Share code block: shown whenever codes exist, even when host is inactive -->
			{#if webrtcStore.hasHostCodes}
				<div class="rounded-lg border border-border bg-muted/40 p-4">
					<p class="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
						Share code
					</p>
					<p class="break-all font-mono text-sm tracking-widest select-all">
						{formatCode(webrtcStore.shareCode)}
					</p>
					<p class="mt-2 text-xs text-muted-foreground">
						Share this 40-character code with remote devices. It includes both the room ID and the
						passcode.
					</p>
				</div>

				<div class="flex flex-wrap gap-2">
					<Button variant="outline" size="sm" onclick={copyCode} class="gap-1.5">
						{#if codeCopied}
							<Check class="h-3.5 w-3.5" />
							Copied
						{:else}
							<Copy class="h-3.5 w-3.5" />
							Copy code
						{/if}
					</Button>

					<Button
						variant="outline"
						size="sm"
						onclick={() => (showRegenerateDialog = true)}
						disabled={regenerating}
						class="gap-1.5"
					>
						{#if regenerating}
							<Loader2 class="h-3.5 w-3.5 animate-spin" />
							Regenerating...
						{:else}
							<RefreshCw class="h-3.5 w-3.5" />
							Regenerate code
						{/if}
					</Button>
				</div>
			{/if}

			<!-- Status row (only when host is active) -->
			{#if webrtcStore.mode === 'host'}
				<div class="flex items-center gap-3">
					{#if webrtcStore.status === 'connecting'}
						<Badge variant="secondary" class="gap-1.5">
							<Loader2 class="h-3 w-3 animate-spin" />
							Connecting to trackers...
						</Badge>
					{:else if webrtcStore.status === 'connected'}
						<Badge variant="default" class="gap-1.5">
							<Wifi class="h-3 w-3" />
							Active
						</Badge>
						<span class="flex items-center gap-1 text-sm text-muted-foreground">
							<Users class="h-3.5 w-3.5" />
							{webrtcStore.peerCount}
							{webrtcStore.peerCount === 1 ? 'client' : 'clients'} connected
						</span>
					{:else if webrtcStore.status === 'error'}
						<Badge variant="destructive" class="gap-1.5">
							<AlertCircle class="h-3 w-3" />
							Error
						</Badge>
						<span class="text-sm text-destructive">{webrtcStore.errorMessage}</span>
					{/if}
				</div>
			{/if}

			<!-- Enable / Disable button -->
			{#if webrtcStore.mode === 'off' || webrtcStore.mode === 'client'}
				<Button
					variant="outline"
					onclick={handleStartHost}
					disabled={webrtcStore.mode === 'client'}
				>
					<Wifi class="h-4 w-4" />
					Enable remote access
				</Button>
			{:else}
				<Button variant="outline" onclick={handleStopHost}>
					<WifiOff class="h-4 w-4" />
					Disable remote access
				</Button>
			{/if}

			{#if webrtcStore.mode === 'client'}
				<p class="text-xs text-muted-foreground">
					Disable join mode first before enabling host mode.
				</p>
			{/if}
		</div>
	</SettingsGroup>

	<!-- ------------------------------------------------------------------ -->
	<!-- CLIENT / JOIN -->
	<!-- ------------------------------------------------------------------ -->
	<SettingsGroup title="Join">
		<div class="space-y-4">
			<p class="text-sm text-muted-foreground">
				Connect to a remote llama.cpp instance. All requests will be routed through the
				peer-to-peer tunnel.
			</p>

			{#if webrtcStore.mode === 'client'}
				<div class="space-y-3">
					<div class="flex items-center gap-3">
						{#if webrtcStore.status === 'connecting'}
							<Badge variant="secondary" class="gap-1.5">
								<Loader2 class="h-3 w-3 animate-spin" />
								Connecting...
							</Badge>
						{:else if webrtcStore.status === 'connected'}
							<Badge variant="default" class="gap-1.5">
								<Wifi class="h-3 w-3" />
								Connected to host
							</Badge>
						{:else if webrtcStore.status === 'error'}
							<Badge variant="destructive" class="gap-1.5">
								<AlertCircle class="h-3 w-3" />
								Disconnected
							</Badge>
							<span class="text-sm text-destructive">{webrtcStore.errorMessage}</span>
						{/if}
					</div>

					<Button variant="outline" onclick={handleLeave}>
						<WifiOff class="h-4 w-4" />
						Leave
					</Button>
				</div>
			{:else}
				<div class="space-y-3">
					<div class="space-y-1.5">
						<label for="join-code" class="text-sm font-medium">Access code</label>
						<Input
							id="join-code"
							placeholder="Paste the 40-character code from the host"
							bind:value={joinInput}
							disabled={joining || webrtcStore.mode === 'host'}
							class="font-mono"
						/>
						{#if joinError}
							<p class="text-sm text-destructive">{joinError}</p>
						{/if}
					</div>

					<Button
						onclick={handleJoin}
						disabled={joining || joinInput.trim().length < 40 || webrtcStore.mode === 'host'}
					>
						{#if joining}
							<Loader2 class="h-4 w-4 animate-spin" />
							Connecting...
						{:else}
							<Wifi class="h-4 w-4" />
							Connect
						{/if}
					</Button>

					{#if webrtcStore.mode === 'host'}
						<p class="text-xs text-muted-foreground">
							Disable host mode first before joining a remote server.
						</p>
					{/if}
				</div>
			{/if}
		</div>
	</SettingsGroup>

	<p class="text-xs text-muted-foreground">
		Uses WebRTC with Google STUN servers for NAT traversal. Signaling via public WebTorrent
		trackers. No data is routed through any relay server.
	</p>
</div>

<!-- Regenerate code confirmation dialog -->
<AlertDialog.Root bind:open={showRegenerateDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>Regenerate access code?</AlertDialog.Title>
			<AlertDialog.Description>
				This will create a new room and passcode. Any devices using the current code will be
				disconnected and will need to be updated with the new code.
			</AlertDialog.Description>
		</AlertDialog.Header>
		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
			<AlertDialog.Action onclick={handleRegenerateConfirm}>Regenerate</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
