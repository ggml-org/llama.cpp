<script lang="ts">
	import * as Alert from '@ggml-org/llama-ui/ui/alert';
	import { AlertTriangle, Loader2, RefreshCw } from '@lucide/svelte';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';
	import { serverStore } from '#lib/stores/index.ts';

	let hasError = $derived(!!serverStore.error);
	let isLoadingModel = $derived(serverStore.status === 503);
</script>

{#if hasError || isLoadingModel}
	<div class="pointer-events-auto mx-auto mb-4 max-w-[48rem] px-1">
		<Alert.Root variant={isLoadingModel ? 'default' : 'destructive'}>
			{#if isLoadingModel}
				<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />
			{:else}
				<AlertTriangle class={ICON_CLASS_DEFAULT} />
			{/if}

			<Alert.Title class="flex items-center justify-between">
				<span>{isLoadingModel ? 'Loading model' : 'Server unavailable'}</span>

				{#if !isLoadingModel}
					<button
						class="flex items-center gap-1.5 rounded-lg bg-destructive/20 px-2 py-1 text-xs font-medium hover:bg-destructive/30 disabled:opacity-50"
						disabled={serverStore.loading}
						onclick={() => serverStore.fetch()}
					>
						<RefreshCw class="size-icon-xs {serverStore.loading ? 'animate-spin' : ''}" />
						{serverStore.loading ? 'Retrying...' : 'Retry'}
					</button>
				{/if}
			</Alert.Title>

			{#if !isLoadingModel}
				<Alert.Description>{serverStore.error}</Alert.Description>
			{/if}
		</Alert.Root>
	</div>
{/if}
