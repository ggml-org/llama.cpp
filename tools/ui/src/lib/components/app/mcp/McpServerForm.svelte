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

	// Detect Authorization by key presence alone, never by value: mid-edit rows
	// may have an empty value, and an empty Authorization row must remain
	// reachable so it can be filled in or cleared via the toggle below.
	let hasAuthorization = $derived(
		headerPairs.some((p) => p.key.trim().toLowerCase() === AUTHORIZATION_HEADER.toLowerCase())
	);

	let wantsAuthorization = $state(false);

	let showAuthorization = $derived(hasAuthorization || wantsAuthorization);

	let urlInput: HTMLInputElement | null = $state(null);
	let authorizationInput: HTMLInputElement | null = $state(null);

	$effect(() => {
		urlInput?.focus();
	});

	$effect(() => {
		if (wantsAuthorization && authorizationInput) {
			authorizationInput.focus();
		}
	});

	// Surface the Authorization header's value verbatim so the exact string that
	// was persisted round-trips on save, whether it carries a "Bearer " prefix
	// or uses a different scheme (e.g. Basic, raw token).
	let authorizationValue = $derived.by(() => {
		const auth = headerPairs.find(
			(p) => p.key.trim().toLowerCase() === AUTHORIZATION_HEADER.toLowerCase()
		);
		return auth?.value ?? '';
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

	function updateAuthorizationValue(value: string) {
		const filtered = headerPairs.filter(
			(p) => p.key.trim().toLowerCase() !== AUTHORIZATION_HEADER.toLowerCase()
		);

		const trimmed = value.trim();

		if (trimmed) {
			filtered.push({ key: AUTHORIZATION_HEADER, value: trimmed });
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
			bind:ref={urlInput}
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
		<div class="mt-2">
			<Input
				id="authorization-{id}"
				type="password"
				autocomplete="off"
				placeholder="e.g. Bearer eyJhbGci..."
				value={authorizationValue}
				oninput={(e) => updateAuthorizationValue(e.currentTarget.value)}
				bind:ref={authorizationInput}
			/>
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
