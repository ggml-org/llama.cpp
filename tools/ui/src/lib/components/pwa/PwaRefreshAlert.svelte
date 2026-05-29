<script lang="ts">
	export let needRefresh: boolean | undefined;
	export let updateServiceWorker: (() => void) | undefined;
</script>

{#if needRefresh && updateServiceWorker}
	<div
		class="fixed right-4 bottom-4 z-[9999] max-w-sm rounded-lg border border-zinc-200 bg-white p-4 shadow-lg dark:border-zinc-700 dark:bg-zinc-800"
	>
		<p class="mb-2 text-sm font-medium">Update available</p>
		<p class="mb-3 text-xs text-zinc-500 dark:text-zinc-400">
			A new version is available. Reload to update.
		</p>
		<div class="flex gap-2">
			<button
				onclick={() => {
					updateServiceWorker();
					needRefresh = false;
				}}
				class="rounded bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700"
			>
				Reload
			</button>
		</div>
	</div>
{/if}
