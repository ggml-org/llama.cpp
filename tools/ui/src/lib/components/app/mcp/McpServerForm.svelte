<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Switch } from '$lib/components/ui/switch';
	import { KeyValuePairs } from '$lib/components/app';
	import type { KeyValuePair } from '$lib/types';
	import { parseHeadersToArray, serializeHeaders } from '$lib/utils';
	import { UrlProtocol } from '$lib/enums';
	import { MCP_SERVER_URL_PLACEHOLDER } from '$lib/constants';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { CLI_FLAGS } from '$lib/constants';

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
		url.toLowerCase().startsWith(UrlProtocol.WEBSOCKET) ||
			url.toLowerCase().startsWith(UrlProtocol.WEBSOCKET_SECURE)
	);

	let headerPairs = $derived<KeyValuePair[]>(parseHeadersToArray(headers));

	const AUTHORIZATION_HEADER = 'Authorization';
	const BEARER_PREFIX = 'Bearer ';

	let hasAuthorization = $derived(
		headerPairs.some(
			(p) => p.key.trim().toLowerCase() === AUTHORIZATION_HEADER.toLowerCase() && p.value.trim()
		)
	);

	let wantsAuthorization = $state(false);

	let showAuthorization = $derived(hasAuthorization || wantsAuthorization);

	let bearerToken = $derived.by(() => {
		const auth = headerPairs.find(
			(p) => p.key.trim().toLowerCase() === AUTHORIZATION_HEADER.toLowerCase()
		);
		if (!auth) return '';

		const value = auth.value.trim();

		if (value.toLowerCase().startsWith(BEARER_PREFIX.toLowerCase())) {
			return value.slice(BEARER_PREFIX.length).trim();
		}

		return value;
	});

	$effect(() => {
		if (!headers.trim()) {
			wantsAuthorization = false;
		}
	});

	function updateHeaderPairs(newPairs: KeyValuePair[]) {
		headerPairs = newPairs;
		onHeadersChange(serializeHeaders(newPairs));
	}

	function updateBearerToken(token: string) {
		const filtered = headerPairs.filter(
			(p) => p.key.trim().toLowerCase() !== AUTHORIZATION_HEADER.toLowerCase()
		);

		const trimmed = token.trim();

		if (trimmed) {
			filtered.push({ key: AUTHORIZATION_HEADER, value: `${BEARER_PREFIX}${trimmed}` });
		}

		updateHeaderPairs(filtered);
	}

	function setUseAuthorization(checked: boolean) {
		wantsAuthorization = checked;

		if (!checked) {
			const filtered = headerPairs.filter(
				(p) => p.key.trim().toLowerCase() !== AUTHORIZATION_HEADER.toLowerCase()
			);
			updateHeaderPairs(filtered);
		}
	}
</script>

<div class="grid gap-2">
	<div class="mb-4">
		<label for="server-url-{id}" class="mb-2 block text-xs font-medium">
			Server URL <span class="text-destructive">*</span>
		</label>

		<Input
			id="server-url-{id}"
			type="url"
			placeholder={MCP_SERVER_URL_PLACEHOLDER}
			value={url}
			oninput={(e) => onUrlChange(e.currentTarget.value)}
			class={urlError ? 'border-destructive' : ''}
		/>

		{#if urlError}
			<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
		{/if}
	</div>

	<label class="flex items-center gap-2 cursor-pointer">
		<Switch
			id="use-authorization-{id}"
			checked={showAuthorization}
			onCheckedChange={setUseAuthorization}
		/>

		<span class="text-xs text-muted-foreground">Authorization</span>
	</label>

	{#if showAuthorization}
		<div class="relative mt-2">
			<Input
				id="bearer-token-{id}"
				type="password"
				autocomplete="off"
				placeholder="paste token here..."
				value={bearerToken}
				oninput={(e) => updateBearerToken(e.currentTarget.value)}
				class="pl-16"
			/>

			<span
				class="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-muted-foreground"
			>
				Bearer
			</span>
		</div>
	{/if}

	<KeyValuePairs
		class="mt-3"
		pairs={headerPairs.filter(
			(p) => p.key.trim().toLowerCase() !== AUTHORIZATION_HEADER.toLowerCase()
		)}
		onPairsChange={(pairs) => {
			const auth = headerPairs.find(
				(p) => p.key.trim().toLowerCase() === AUTHORIZATION_HEADER.toLowerCase()
			);
			updateHeaderPairs(auth ? [...pairs, auth] : pairs);
		}}
		keyPlaceholder="Header name"
		valuePlaceholder="Value"
		addButtonLabel="Add"
		emptyMessage="No custom headers configured."
		sectionLabel="Custom Headers"
		sectionLabelOptional
	/>

	{#if !isWebSocket && onUseProxyChange}
		<label
			class={[
				'mt-3 flex items-start gap-2',
				mcpStore.isProxyAvailable && 'cursor-pointer',
				!mcpStore.isProxyAvailable && 'opacity-80'
			]}
		>
			<Switch
				class="mt-1"
				id="use-proxy-{id}"
				checked={useProxy}
				disabled={!mcpStore.isProxyAvailable}
				onCheckedChange={(checked) => onUseProxyChange?.(checked)}
			/>

			<span>
				<span class="text-xs text-muted-foreground">Use llama-server proxy</span>

				<br />

				{#if !mcpStore.isProxyAvailable}
					<span class="inline-flex gap-0.75 text-xs text-muted-foreground/60"
						>(Run <pre>llama-server</pre>
						with
						<pre>{CLI_FLAGS.MCP_PROXY}</pre>
						flag)</span
					>
				{/if}
			</span>
		</label>
	{/if}
</div>
