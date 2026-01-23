<script lang="ts">
	import { base } from '$app/paths';
	import { AlertTriangle, RefreshCw, Key, CheckCircle, XCircle } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { serverStore, serverLoading } from '$lib/stores/server.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { t } from '$lib/i18n';
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

	let isServerLoading = $derived(serverLoading());
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
		const currentConfig = config();
		apiKeyInput = currentConfig.apiKey?.toString() || '';
	}

	async function handleSaveApiKey() {
		if (!apiKeyInput.trim()) return;

		apiKeyState = 'validating';
		apiKeyError = '';

		try {
			// Update the API key in settings first
			settingsStore.updateConfig('apiKey', apiKeyInput.trim());

			// Test the API key by making a real request to the server
			const response = await fetch(`${base}/props`, {
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${apiKeyInput.trim()}`
				}
			});

			if (response.ok) {
				// API key is valid - User Story B
				apiKeyState = 'success';

				// Show success state briefly, then navigate to home
				setTimeout(() => {
					goto(`#/`);
				}, 1000);
			} else {
				// API key is invalid - User Story A
				apiKeyState = 'error';

				if (response.status === 401 || response.status === 403) {
					apiKeyError = t('server.error.api_key.invalid');
				} else {
					apiKeyError = t('server.error.api_key.auth_failed', { status: response.status });
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
					apiKeyError = t('server.error.api_key.cannot_connect');
				} else {
					apiKeyError = t('server.error.api_key.generic_error', { message: error.message });
				}
			} else {
				apiKeyError = t('server.error.api_key.connection_error');
			}

			// Reset to idle state after showing error (don't reload UI)
			setTimeout(() => {
				apiKeyState = 'idle';
			}, 3000);
		}
	}

	function handleApiKeyKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter') {
			handleSaveApiKey();
		}
	}
</script>

<div class="flex h-full items-center justify-center {className}">
	<div class="w-full max-w-md px-4 text-center">
		<div class="mb-6" in:fade={{ duration: 300 }}>
			<div
				class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10"
			>
				<AlertTriangle class="h-8 w-8 text-destructive" />
			</div>

			<h2 class="mb-2 text-xl font-semibold">{t('server.error.title')}</h2>

			<p class="mb-4 text-sm text-muted-foreground">
				{error}
			</p>
		</div>

		{#if isAccessDeniedError && !showApiKeyInput}
			<div in:fly={{ y: 10, duration: 300, delay: 200 }} class="mb-4">
				<Button onclick={handleShowApiKeyInput} variant="outline" class="w-full">
					<Key class="h-4 w-4" />
					{t('server.error.api_key.enter')}
				</Button>
			</div>
		{/if}

		{#if showApiKeyInput}
			<div in:fly={{ y: 10, duration: 300, delay: 200 }} class="mb-4 space-y-3 text-left">
				<div class="space-y-2">
					<Label for="api-key-input" class="text-sm font-medium">
						{t('server.error.api_key.label')}
					</Label>

					<div class="relative">
						<Input
							id="api-key-input"
							placeholder={t('server.error.api_key.placeholder')}
							bind:value={apiKeyInput}
							onkeydown={handleApiKeyKeydown}
							class="w-full pr-10 {apiKeyState === 'error'
								? 'border-destructive'
								: apiKeyState === 'success'
									? 'border-green-500'
									: ''}"
							disabled={apiKeyState === 'validating'}
						/>
						{#if apiKeyState === 'validating'}
							<div class="absolute top-1/2 right-3 -translate-y-1/2">
								<RefreshCw class="h-4 w-4 animate-spin text-muted-foreground" />
							</div>
						{:else if apiKeyState === 'success'}
							<div
								class="absolute top-1/2 right-3 -translate-y-1/2"
								in:scale={{ duration: 200, start: 0.8 }}
							>
								<CheckCircle class="h-4 w-4 text-green-500" />
							</div>
						{:else if apiKeyState === 'error'}
							<div
								class="absolute top-1/2 right-3 -translate-y-1/2"
								in:scale={{ duration: 200, start: 0.8 }}
							>
								<XCircle class="h-4 w-4 text-destructive" />
							</div>
						{/if}
					</div>
					{#if apiKeyError}
						<p class="text-sm text-destructive" in:fly={{ y: -10, duration: 200 }}>
							{apiKeyError}
						</p>
					{/if}
					{#if apiKeyState === 'success'}
						<p class="text-sm text-green-600" in:fly={{ y: -10, duration: 200 }}>
							<span aria-hidden="true">✓</span> {t('server.error.api_key.success')}
						</p>
					{/if}
				</div>
				<div class="flex gap-2">
					<Button
						onclick={handleSaveApiKey}
						disabled={!apiKeyInput.trim() ||
							apiKeyState === 'validating' ||
							apiKeyState === 'success'}
						class="flex-1"
					>
						{#if apiKeyState === 'validating'}
							<RefreshCw class="h-4 w-4 animate-spin" />
							{t('server.error.api_key.validating')}
						{:else if apiKeyState === 'success'}
							{t('server.error.api_key.success_button')}
						{:else}
							{t('server.error.api_key.save_retry')}
						{/if}
					</Button>
					<Button
						onclick={() => {
							showApiKeyInput = false;
							apiKeyState = 'idle';
							apiKeyError = '';
						}}
						variant="outline"
						class="flex-1"
						disabled={apiKeyState === 'validating'}
					>
						{t('server.error.api_key.cancel')}
					</Button>
				</div>
			</div>
		{/if}

		{#if showRetry}
			<div in:fly={{ y: 10, duration: 300, delay: 200 }}>
				<Button onclick={handleRetryConnection} disabled={isServerLoading} class="w-full">
					{#if isServerLoading}
						<RefreshCw class="h-4 w-4 animate-spin" />

						{t('server.error.retry.connecting')}
					{:else}
						<RefreshCw class="h-4 w-4" />

						{t('server.error.retry.action')}
					{/if}
				</Button>
			</div>
		{/if}

		{#if showTroubleshooting}
			<div class="mt-4 text-left" in:fly={{ y: 10, duration: 300, delay: 400 }}>
				<details class="text-sm">
					<summary class="cursor-pointer text-muted-foreground hover:text-foreground">
						{t('server.error.troubleshooting.title')}
					</summary>

					<div class="mt-2 space-y-3 text-xs text-muted-foreground">
						<div class="space-y-2">
							<p class="mb-4 font-medium">{t('server.error.troubleshooting.start')}</p>

							<div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
								<p>{t('server.error.troubleshooting.command_hf')}</p>
							</div>

							<p>{t('server.error.troubleshooting.or')}</p>

							<div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
								<p class="mt-1">{t('server.error.troubleshooting.command_local')}</p>
							</div>
						</div>
						<ul class="list-disc space-y-1 pl-4">
							<li>{t('server.error.troubleshooting.step_url')}</li>

							<li>{t('server.error.troubleshooting.step_network')}</li>

							<li>{t('server.error.troubleshooting.step_logs')}</li>
						</ul>
					</div>
				</details>
			</div>
		{/if}
	</div>
</div>
