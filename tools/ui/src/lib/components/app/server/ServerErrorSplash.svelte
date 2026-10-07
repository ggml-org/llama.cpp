<script lang="ts">
	import { AlertTriangle, CheckCircle, Key, RefreshCw, XCircle } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { HEADERS, ICON_CLASS_DEFAULT, ROUTES, SETTINGS_KEYS } from '$lib/constants';
	import { KeyboardKey } from '$lib/enums';
	import { serverStore, settingsStore } from '$lib/stores';
	import { fade, fly, scale } from 'svelte/transition';

	interface Props {
		class?: string;
		error: string;
		onRetry?: () => void;
		showRetry?: boolean;
		showTroubleshooting?: boolean;
	}

	let {
		class: className = '',
		error,
		onRetry,
		showRetry = true,
		showTroubleshooting = false
	}: Props = $props();

	let isServerLoading = $derived(serverStore.loading);
	let isAccessDeniedError = $derived(
		error.toLowerCase().includes('access denied') ||
			error.toLowerCase().includes('invalid api key') ||
			error.toLowerCase().includes('unauthorized') ||
			error.toLowerCase().includes('401') ||
			error.toLowerCase().includes('403')
	);

	let apiKeyInput = $state('');
	let showApiKeyInput = $state(false);
	let apiKeyState = $state<'idle' | 'validating' | 'success' | 'error'>('idle');
	let apiKeyError = $state('');

	function handleRetryConnection() {
		if (onRetry) {
			onRetry();
		} else {
			serverStore.fetch();
		}
	}

	function handleShowApiKeyInput() {
		showApiKeyInput = true;
		// Pre-fill with current API key if it exists
		const currentConfig = settingsStore.config;

		apiKeyInput = currentConfig.apiKey?.toString() || '';
	}

	async function handleSaveApiKey() {
		if (!apiKeyInput.trim()) return;

		apiKeyState = 'validating';
		apiKeyError = '';

		try {
			// Update the API key in settings first
			settingsStore.updateConfig(SETTINGS_KEYS.API_KEY, apiKeyInput.trim());

			// Test the API key by making a real request to the server
			const response = await fetch(`${base}/props`, {
				headers: {
					'Content-Type': 'application/json',
					[HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${apiKeyInput.trim()}`
				}
			});

			if (response.ok) {
				// API key is valid - User Story B
				apiKeyState = 'success';

				// Show success state briefly, then navigate to home
				setTimeout(() => {
					goto(ROUTES.START);
				}, 1000);
			} else {
				// API key is invalid - User Story A
				apiKeyState = 'error';

				if (response.status === 401 || response.status === 403) {
					apiKeyError = 'Invalid API key - please check and try again';
				} else {
					apiKeyError = `Authentication failed (${response.status})`;
				}

				// Reset to idle state after showing error (don't reload UI)
				setTimeout(() => {
					apiKeyState = 'idle';
				}, 3000);
			}
		} catch (error) {
			// Network or other errors - User Story A
			apiKeyState = 'error';

			if (error instanceof Error) {
				if (error.message.includes('fetch')) {
					apiKeyError = 'Cannot connect to server - check if server is running';
				} else {
					apiKeyError = error.message;
				}
			} else {
				apiKeyError = 'Connection error - please try again';
			}

			// Reset to idle state after showing error (don't reload UI)
			setTimeout(() => {
				apiKeyState = 'idle';
			}, 3000);
		}
	}

	function handleApiKeyKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER) {
			handleSaveApiKey();
		}
	}
</script>

<div class="flex h-full items-center justify-center {className}">
	<div class="w-full max-w-md px-4 text-center">
		<div in:fade={{ duration: 300 }} class="mb-6">
			<div
				class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10"
			>
				<AlertTriangle class="h-8 w-8 text-destructive" />
			</div>

			<h2 class="mb-2 text-xl font-semibold">Server Connection Error</h2>

			<p class="mb-4 text-sm text-muted-foreground">
				{error}
			</p>
		</div>

		{#if isAccessDeniedError && !showApiKeyInput}
			<div in:fly={{ delay: 200, duration: 300, y: 10 }} class="mb-4">
				<Button class="w-full" onclick={handleShowApiKeyInput} variant="outline">
					<Key class={ICON_CLASS_DEFAULT} />
					Enter API Key
				</Button>
			</div>
		{/if}

		{#if showApiKeyInput}
			<div in:fly={{ delay: 200, duration: 300, y: 10 }} class="mb-4 space-y-3 text-left">
				<div class="space-y-2">
					<Label class="text-sm font-medium" for="api-key-input">API Key</Label>

					<div class="relative">
						<Input
							bind:value={apiKeyInput}
							autocomplete="new-password"
							class="w-full pr-10 {apiKeyState === 'error'
								? 'border-destructive'
								: apiKeyState === 'success'
									? 'border-green-500'
									: ''}"
							disabled={apiKeyState === 'validating'}
							id="api-key-input"
							onkeydown={handleApiKeyKeydown}
							placeholder="Enter your API key..."
							type="password"
						/>

						{#if apiKeyState === 'validating'}
							<div class="absolute top-1/2 right-3 -translate-y-1/2">
								<RefreshCw class="{ICON_CLASS_DEFAULT} animate-spin text-muted-foreground" />
							</div>
						{:else if apiKeyState === 'success'}
							<div
								in:scale={{ duration: 200, start: 0.8 }}
								class="absolute top-1/2 right-3 -translate-y-1/2"
							>
								<CheckCircle class="{ICON_CLASS_DEFAULT} text-green-500" />
							</div>
						{:else if apiKeyState === 'error'}
							<div
								in:scale={{ duration: 200, start: 0.8 }}
								class="absolute top-1/2 right-3 -translate-y-1/2"
							>
								<XCircle class="{ICON_CLASS_DEFAULT} text-destructive" />
							</div>
						{/if}
					</div>

					{#if apiKeyError}
						<p in:fly={{ duration: 200, y: -10 }} class="text-sm text-destructive">
							{apiKeyError}
						</p>
					{/if}

					{#if apiKeyState === 'success'}
						<p in:fly={{ duration: 200, y: -10 }} class="text-sm text-green-600">
							✓ API key validated successfully! Connecting...
						</p>
					{/if}
				</div>

				<div class="flex gap-2">
					<Button
						class="flex-1"
						disabled={!apiKeyInput.trim() ||
							apiKeyState === 'validating' ||
							apiKeyState === 'success'}
						onclick={handleSaveApiKey}
					>
						{#if apiKeyState === 'validating'}
							<RefreshCw class="{ICON_CLASS_DEFAULT} animate-spin" />
							Validating...
						{:else if apiKeyState === 'success'}
							Success!
						{:else}
							Save & Retry
						{/if}
					</Button>

					<Button
						class="flex-1"
						disabled={apiKeyState === 'validating'}
						onclick={() => {
							showApiKeyInput = false;
							apiKeyState = 'idle';
							apiKeyError = '';
						}}
						variant="outline"
					>
						Cancel
					</Button>
				</div>
			</div>
		{/if}

		{#if showRetry}
			<div in:fly={{ delay: 200, duration: 300, y: 10 }}>
				<Button class="w-full" disabled={isServerLoading} onclick={handleRetryConnection}>
					{#if isServerLoading}
						<RefreshCw class="{ICON_CLASS_DEFAULT} animate-spin" />

						Connecting...
					{:else}
						<RefreshCw class={ICON_CLASS_DEFAULT} />

						Retry Connection
					{/if}
				</Button>
			</div>
		{/if}

		{#if showTroubleshooting}
			<div in:fly={{ delay: 400, duration: 300, y: 10 }} class="mt-4 text-left">
				<details class="text-sm">
					<summary class="cursor-pointer text-muted-foreground hover:text-foreground">
						Troubleshooting
					</summary>

					<div class="mt-2 space-y-3 text-xs text-muted-foreground">
						<div class="space-y-2">
							<p class="mb-4 font-medium">Start the llama-server:</p>

							<div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
								<p>llama-server -hf ggml-org/gemma-3-4b-it-GGUF</p>
							</div>

							<p>or</p>

							<div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
								<p class="mt-1">llama-server -m locally-stored-model.gguf</p>
							</div>
						</div>

						<ul class="list-disc space-y-1 pl-4">
							<li>Check that the server is accessible at the correct URL</li>

							<li>Verify your network connection</li>

							<li>Check server logs for any error messages</li>
						</ul>
					</div>
				</details>
			</div>
		{/if}
	</div>
</div>
