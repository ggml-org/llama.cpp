<script lang="ts">
	import { Plus, X } from '@lucide/svelte';
	import { Input } from '$lib/components/ui/input';
	import { autoResizeTextarea } from '$lib/utils';
	import { type HeaderPair, parseHeadersToArray, serializeHeaders } from '$lib/utils/mcp';

	interface Props {
		url: string;
		headers: string;
		onUrlChange: (url: string) => void;
		onHeadersChange: (headers: string) => void;
		urlError?: string | null;
		id?: string;
	}

	let {
		url,
		headers,
		onUrlChange,
		onHeadersChange,
		urlError = null,
		id = 'server'
	}: Props = $props();

	// Local state for header pairs
	let headerPairs = $state<HeaderPair[]>(parseHeadersToArray(headers));

	// Sync header pairs to parent when they change
	function updateHeaderPairs(newPairs: HeaderPair[]) {
		headerPairs = newPairs;
		onHeadersChange(serializeHeaders(newPairs));
	}

	function addHeaderPair() {
		updateHeaderPairs([...headerPairs, { key: '', value: '' }]);
	}

	function removeHeaderPair(index: number) {
		updateHeaderPairs(headerPairs.filter((_, i) => i !== index));
	}

	function updatePairKey(index: number, key: string) {
		const newPairs = [...headerPairs];
		newPairs[index] = { ...newPairs[index], key };
		updateHeaderPairs(newPairs);
	}

	function updatePairValue(index: number, value: string) {
		const newPairs = [...headerPairs];
		newPairs[index] = { ...newPairs[index], value };
		updateHeaderPairs(newPairs);
	}
</script>

<div class="space-y-3">
	<div>
		<label for="server-url-{id}" class="mb-1 block text-xs font-medium">
			Server URL <span class="text-destructive">*</span>
		</label>
		<Input
			id="server-url-{id}"
			type="url"
			placeholder="https://mcp.example.com/sse"
			value={url}
			oninput={(e) => onUrlChange(e.currentTarget.value)}
			class={urlError ? 'border-destructive' : ''}
		/>
		{#if urlError}
			<p class="mt-1 text-xs text-destructive">{urlError}</p>
		{/if}
	</div>

	<div>
		<div class="mb-1 flex items-center justify-between">
			<span class="text-xs font-medium">
				Custom Headers <span class="text-muted-foreground">(optional)</span>
			</span>

			<button
				type="button"
				class="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
				onclick={addHeaderPair}
			>
				<Plus class="h-3 w-3" />
				Add
			</button>
		</div>
		{#if headerPairs.length > 0}
			<div class="space-y-2">
				{#each headerPairs as pair, index (index)}
					<div class="flex items-start gap-2">
						<Input
							type="text"
							placeholder="Header name"
							value={pair.key}
							oninput={(e) => updatePairKey(index, e.currentTarget.value)}
							class="flex-1"
						/>
						<textarea
							placeholder="Value"
							value={pair.value}
							oninput={(e) => {
								updatePairValue(index, e.currentTarget.value);
								autoResizeTextarea(e.currentTarget);
							}}
							class="flex-1 resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm leading-5 placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:outline-none"
							rows="1"
						></textarea>
						<button
							type="button"
							class="shrink-0 p-1 text-muted-foreground hover:text-destructive"
							onclick={() => removeHeaderPair(index)}
							aria-label="Remove header"
						>
							<X class="h-3.5 w-3.5" />
						</button>
					</div>
				{/each}
			</div>
		{:else}
			<p class="text-xs text-muted-foreground">No custom headers configured.</p>
		{/if}
	</div>
</div>
