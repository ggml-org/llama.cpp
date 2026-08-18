<script lang="ts">
	import { AlertCircle, Loader2, Wifi, WifiOff } from '@lucide/svelte';
	import { SettingsGroup } from '$lib/components/app/settings';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { CODE_LENGTHS } from '$lib/constants';
	import { RemoteAccessMode, TunnelStatus } from '$lib/enums';
	import { modelsStore, serverStore } from '$lib/stores';
	import { webrtcStore } from '$lib/stores/webrtc.svelte';
	import { fade } from 'svelte/transition';

	let joinInput = $state('');
	let joinError = $state('');
	// The store owns the connection state, so leaving mid-attempt re-enables
	// the form right away.
	let connecting = $derived(webrtcStore.status === TunnelStatus.CONNECTING);

	// Models and props come from whichever backend served them, so they are
	// refetched when the active backend changes.
	async function refreshBackendState() {
		await serverStore.fetch();
		await modelsStore.fetch(true);
	}

	async function handleReconnect() {
		try {
			await webrtcStore.reconnect();
			await refreshBackendState();
		} catch {
			// state is already reflected in the store
		}
	}

	async function handleJoin() {
		joinError = '';
		const code = joinInput.trim().replace(/\s/g, '');

		if (code.length < CODE_LENGTHS.SHARE) {
			joinError = `Code must be ${CODE_LENGTHS.SHARE} characters`;

			return;
		}

		try {
			await webrtcStore.joinAsClient(code);
			await refreshBackendState();
		} catch (e) {
			joinError = e instanceof Error ? e.message : String(e);
		}
	}

	async function handleLeave() {
		webrtcStore.leaveAsClient();
		joinInput = '';
		joinError = '';
		await refreshBackendState();
	}
</script>

<div class="space-y-12" in:fade={{ duration: 150 }}>
	<SettingsGroup title="Join">
		<div class="space-y-4">
			<p class="text-sm text-muted-foreground">
				Connect to a remote llama.cpp instance running llama-connect. All requests will be routed
				through the peer-to-peer tunnel.
			</p>

			{#if webrtcStore.mode === RemoteAccessMode.CLIENT}
				<div class="space-y-3">
					<div class="flex items-center gap-3">
						{#if webrtcStore.status === TunnelStatus.CONNECTING}
							<Badge variant="secondary" class="gap-1.5">
								<Loader2 class="h-3 w-3 animate-spin" />
								Connecting...
							</Badge>
						{:else if webrtcStore.status === TunnelStatus.CONNECTED}
							<Badge variant="default" class="gap-1.5">
								<Wifi class="h-3 w-3" />
								Connected to host
							</Badge>
						{:else if webrtcStore.status === TunnelStatus.ERROR}
							<Badge variant="destructive" class="gap-1.5">
								<AlertCircle class="h-3 w-3" />
								Disconnected
							</Badge>
							<span class="text-sm text-destructive">{webrtcStore.errorMessage}</span>
						{/if}
					</div>

					{#if webrtcStore.status === TunnelStatus.ERROR}
						<p class="text-sm text-muted-foreground">
							Requests stay blocked while remote access is on, so nothing is sent to the server
							hosting this page. Reconnect to the remote instance, or leave to use this server.
						</p>
					{/if}

					<div class="flex flex-wrap gap-2">
						{#if webrtcStore.status === TunnelStatus.ERROR}
							<Button variant="outline" onclick={handleReconnect} disabled={connecting}>
								{#if connecting}
									<Loader2 class="h-4 w-4 animate-spin" />
									Reconnecting...
								{:else}
									<Wifi class="h-4 w-4" />
									Reconnect
								{/if}
							</Button>
						{/if}

						<Button variant="outline" onclick={handleLeave}>
							<WifiOff class="h-4 w-4" />
							Leave
						</Button>
					</div>
				</div>
			{:else}
				<div class="space-y-3">
					<div class="space-y-1.5">
						<label for="join-code" class="text-sm font-medium">Access code</label>
						<Input
							id="join-code"
							placeholder="Paste the {CODE_LENGTHS.SHARE}-character code from the host"
							bind:value={joinInput}
							disabled={connecting}
							class="font-mono"
						/>
						{#if joinError}
							<p class="text-sm text-destructive">{joinError}</p>
						{/if}
					</div>

					<Button
						onclick={handleJoin}
						disabled={connecting || joinInput.trim().length < CODE_LENGTHS.SHARE}
					>
						{#if connecting}
							<Loader2 class="h-4 w-4 animate-spin" />
							Connecting...
						{:else}
							<Wifi class="h-4 w-4" />
							Connect
						{/if}
					</Button>
				</div>
			{/if}
		</div>
	</SettingsGroup>

	<p class="text-xs text-muted-foreground">
		Uses WebRTC with Google STUN servers for NAT traversal. Signaling via public WebTorrent
		trackers. No data is routed through any relay server.
	</p>
</div>
