<script lang="ts">
	import { AlertTriangle, RefreshCw } from '@lucide/svelte';
	import { serverLoading, serverStore } from '$lib/stores/server.svelte';
	import { fly } from 'svelte/transition';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();

	function handleRefreshServer() {
		serverStore.fetchServerProps();
	}
</script>

<div class="mb-3 {className}" in:fly={{ y: 10, duration: 250 }}>
	<div
		class="rounded-md border border-yellow-200 bg-yellow-50 px-3 py-2 dark:border-yellow-800 dark:bg-yellow-950"
	>
		<div class="flex items-center justify-between">
			<div class="flex items-center">
				<AlertTriangle class="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
				<p class="ml-2 text-sm text-yellow-800 dark:text-yellow-200">
					Server `/props` endpoint not available - using cached data
				</p>
			</div>
			<button
				onclick={handleRefreshServer}
				disabled={serverLoading()}
				class="ml-3 flex items-center gap-1.5 rounded bg-yellow-100 px-2 py-1 text-xs font-medium text-yellow-800 hover:bg-yellow-200 disabled:opacity-50 dark:bg-yellow-900 dark:text-yellow-200 dark:hover:bg-yellow-800"
			>
				<RefreshCw class="h-3 w-3 {serverLoading() ? 'animate-spin' : ''}" />
				{serverLoading() ? 'Checking...' : 'Retry'}
			</button>
		</div>
	</div>
</div>
