<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Switch } from '$lib/components/ui/switch';
	import { KeyValuePairs } from '$lib/components/app';
	import type { KeyValuePair } from '$lib/types';
	import { parseHeadersToArray, serializeHeaders } from '$lib/utils';

	interface Props {
		url: string;
		headers: string;
		useProxy?: boolean;
		onUrlChange: (url: string) => void;
		onHeadersChange: (headers: string) => void;
		onUseProxyChange?: (useProxy: boolean) => void;
		urlError?: string | null;
		id?: string;
	}

	let {
		url,
		headers,
		useProxy = false,
		onUrlChange,
		onHeadersChange,
		onUseProxyChange,
		urlError = null,
		id = 'server'
	}: Props = $props();

	let isWebSocket = $derived(
		url.toLowerCase().startsWith('ws://') || url.toLowerCase().startsWith('wss://')
	);

	let headerPairs = $derived<KeyValuePair[]>(parseHeadersToArray(headers));

	function updateHeaderPairs(newPairs: KeyValuePair[]) {
		headerPairs = newPairs;
		onHeadersChange(serializeHeaders(newPairs));
	}
</script>

<div class="grid gap-3">
	<div>
		<label for="server-url-{id}" class="mb-2 block text-xs font-medium">
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
			<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
		{/if}

		{#if !isWebSocket && onUseProxyChange}
			<label class="mt-3 flex cursor-pointer items-center gap-2">
				<Switch
					id="use-proxy-{id}"
					checked={useProxy}
					onCheckedChange={(checked) => onUseProxyChange?.(checked)}
				/>

				<span class="text-xs text-muted-foreground">Use llama-server proxy</span>
			</label>
		{/if}
	</div>

	<KeyValuePairs
		class="mt-2"
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
