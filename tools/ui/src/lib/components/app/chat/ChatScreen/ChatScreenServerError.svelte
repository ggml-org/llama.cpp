<script lang="ts">
	import { AlertTriangle, Loader2, RefreshCw } from '@lucide/svelte';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import * as Alert from '$lib/components/ui/alert';
	import { serverError, serverLoading, serverStatus, serverStore } from '$lib/stores/server.svelte';

	let hasError = $derived(!!serverError());
	let isLoadingModel = $derived(serverStatus() === 503);
</script>

{#if hasError}
	<div
		class="pointer-events-auto mx-auto mb-4 max-w-[48rem] px-1"
		use:fadeInView={{ y: 10, duration: 250 }}
	>
		<Alert.Root variant={isLoadingModel ? 'default' : 'destructive'}>
			{#if isLoadingModel}
				<Loader2 class="h-4 w-4 animate-spin" />
			{:else}
				<AlertTriangle class="h-4 w-4" />
			{/if}

			<Alert.Title class="flex items-center justify-between">
				<span>{isLoadingModel ? 'Loading model' : 'Server unavailable'}</span>

				{#if !isLoadingModel}
					<button
						onclick={() => serverStore.fetch()}
						disabled={serverLoading()}
						class="flex items-center gap-1.5 rounded-lg bg-destructive/20 px-2 py-1 text-xs font-medium hover:bg-destructive/30 disabled:opacity-50"
					>
						<RefreshCw class="h-3 w-3 {serverLoading() ? 'animate-spin' : ''}" />
						{serverLoading() ? 'Retrying...' : 'Retry'}
					</button>
				{/if}
			</Alert.Title>

			{#if !isLoadingModel}
				<Alert.Description>{serverError()}</Alert.Description>
			{/if}
		</Alert.Root>
	</div>
{/if}
