<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { KeyValuePairs } from '$lib/components/app';
	import type { KeyValuePair } from '$lib/types';
	import { parseHeadersToArray, serializeHeaders } from '$lib/utils/mcp';

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
	let headerPairs = $state<KeyValuePair[]>(parseHeadersToArray(headers));

	$effect(() => {
		headerPairs = parseHeadersToArray(headers);
	});

	// Sync header pairs to parent when they change
	function updateHeaderPairs(newPairs: KeyValuePair[]) {
		headerPairs = newPairs;
		onHeadersChange(serializeHeaders(newPairs));
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

	<KeyValuePairs
		pairs={headerPairs}
		onPairsChange={updateHeaderPairs}
		keyPlaceholder="Header name"
		valuePlaceholder="Value"
		addButtonLabel="Add"
		emptyMessage="No custom headers configured."
		sectionLabel="Custom Headers"
		sectionLabelOptional={true}
	/>
</div>
