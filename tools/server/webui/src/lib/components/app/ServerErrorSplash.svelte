<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { AlertTriangle, RefreshCw } from '@lucide/svelte';
	import { ServerStatus } from '$lib/components/app';
	import { serverStore, serverLoading } from '$lib/stores/server.svelte';
	import { fade, fly } from 'svelte/transition';

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
		showTroubleshooting = true,
	}: Props = $props();

	const isServerLoading = $derived(serverLoading());

	function handleRetryConnection() {
		if (onRetry) {
			onRetry();
		} else {
			serverStore.fetchServerProps();
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

			<h2 class="mb-2 text-xl font-semibold">Server Connection Error</h2>

			<p class="mb-4 text-sm text-muted-foreground">
				{error}
			</p>

			<div class="mb-4">
				<ServerStatus showActions={true} class="justify-center" />
			</div>
		</div>

		{#if showRetry}
			<div in:fly={{ y: 10, duration: 300, delay: 200 }}>
				<Button onclick={handleRetryConnection} disabled={isServerLoading} class="w-full">
					{#if isServerLoading}
						<RefreshCw class="mr-2 h-4 w-4 animate-spin" />

						Connecting...
					{:else}
						<RefreshCw class="mr-2 h-4 w-4" />

						Retry Connection
					{/if}
				</Button>
			</div>
		{/if}

		{#if showTroubleshooting}
			<div class="mt-4 text-left" in:fly={{ y: 10, duration: 300, delay: 400 }}>
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
