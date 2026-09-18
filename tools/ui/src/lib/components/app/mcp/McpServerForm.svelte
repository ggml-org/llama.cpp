<script lang="ts">
	import { KeyValuePairs } from '$lib/components/app';
	import { Input } from '$lib/components/ui/input';
	import { Switch } from '$lib/components/ui/switch';
	import { CLI_FLAGS, HEADERS, MCP_SERVER_URL_PLACEHOLDER } from '$lib/constants';
	import { UrlProtocol } from '$lib/enums';
	import { mcpStore } from '$lib/stores';
	import type { KeyValuePair } from '$lib/types';
	import { parseHeadersToArray, serializeHeaders } from '$lib/utils';

	interface Props {
		url: string;
		headers: string;
		name?: string;
		onNameChange?: (name: string) => void;
		/** Shown in the empty display name field, e.g. the current automatic label. */
		namePlaceholder?: string;
		useProxy?: boolean;
		onUrlChange: (url: string) => void;
		onHeadersChange: (headers: string) => void;
		onUseProxyChange?: (useProxy: boolean) => void;
		urlError?: string | null;
		id?: string;
		/**
		 * "Wants Authorization" is the user's *intent* to add a Bearer token
		 * (separate from `hasAuthorization` which reflects what's already in
		 * the headers). Bindable so a parent - e.g. the recommendation cards
		 * on the "Add New Server" dialog - can flip the switch on when the
		 * picked server ships a `needsAuthorization: true` flag.
		 */
		wantsAuthorization?: boolean;
		/**
		 * Marks the "Authorization" field as required. Locks the toggle so the
		 * user can't dismiss it, and visually marks the field with a red
		 * asterisk. The parent is expected to gate its submit affordance on
		 * the bearer token actually being filled. Used by the "Add New Server"
		 * dialog for recommendations whose `needsAuthorization` flag is true.
		 */
		required?: boolean;
	}

	let {
		headers,
		id = 'server',
		name = '',
		namePlaceholder = 'Name reported by the server',
		onHeadersChange,
		onNameChange,
		onUrlChange,
		onUseProxyChange,
		required = false,
		url,
		urlError = null,
		useProxy = false,
		wantsAuthorization = $bindable(false)
	}: Props = $props();

	let isWebSocket = $derived(
		url.toLowerCase().startsWith(UrlProtocol.WEBSOCKET) ||
			url.toLowerCase().startsWith(UrlProtocol.WEBSOCKET_SECURE)
	);

	let headerPairs = $derived<KeyValuePair[]>(parseHeadersToArray(headers));

	// Heuristic: this dedicated UI only owns Authorization headers that already
	// carry a Bearer scheme. Anything else (e.g. Basic, raw tokens) stays in the
	// KV section so the user can still edit those values verbatim.
	const matchesAuthorizationKey = (key: string): boolean =>
		HEADERS.REDACTED.has(key.trim().toLowerCase());

	const isBearerScheme = (value: string): boolean =>
		value.trim().toLowerCase().startsWith(HEADERS.BEARER.toLowerCase());

	const ownedByBearerUi = (p: KeyValuePair): boolean =>
		matchesAuthorizationKey(p.key) && isBearerScheme(p.value);

	let hasAuthorization = $derived(headerPairs.some(ownedByBearerUi));

	let showAuthorization = $derived(hasAuthorization || wantsAuthorization);

	let urlInput: HTMLInputElement | null = $state(null);
	let bearerInput: HTMLInputElement | null = $state(null);

	$effect(() => {
		urlInput?.focus();
	});

	$effect(() => {
		if (wantsAuthorization && bearerInput) {
			bearerInput.focus();
		}
	});

	let bearerToken = $derived.by(() => {
		const auth = headerPairs.find(ownedByBearerUi);

		if (!auth) return '';

		return auth.value.trim().slice(HEADERS.BEARER.length).trim();
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

	// The dedicated UI owns the Authorization slot end-to-end when the user
	// engages it: any prior Authorization row (Bearer or otherwise) is replaced
	// by exactly one { Authorization: "Bearer <token>" } entry. JSON's last-key
	// behavior would otherwise pick one arbitrarily, so we strip first.
	function updateBearerToken(token: string) {
		const filtered = headerPairs.filter((p) => !matchesAuthorizationKey(p.key));
		const trimmed = token.trim();

		if (trimmed) {
			filtered.push({ key: HEADERS.AUTHORIZATION, value: `${HEADERS.BEARER}${trimmed}` });
		}

		updateHeaderPairs(filtered);
	}

	function setUseAuthorization(checked: boolean) {
		wantsAuthorization = checked;

		if (!checked) {
			// Only drop the entry this UI owns; a non-Bearer Authorization row
			// authored in the KV section must survive a toggle off untouched.
			const filtered = headerPairs.filter((p) => !ownedByBearerUi(p));

			updateHeaderPairs(filtered);
		}
	}
</script>

<div class="grid gap-2">
	<div class="mb-4">
		<label class="mb-2 block text-xs font-medium select-none" for="server-url-{id}">
			Server URL <span class="text-destructive">*</span>
		</label>

		<Input
			bind:ref={urlInput}
			class={urlError ? 'border-destructive' : ''}
			id="server-url-{id}"
			oninput={(e) => onUrlChange(e.currentTarget.value)}
			placeholder={MCP_SERVER_URL_PLACEHOLDER}
			type="url"
			value={url}
		/>

		{#if urlError}
			<p class="mt-1.5 text-xs text-destructive">{urlError}</p>
		{/if}
	</div>

	<div class="mb-4">
		<label class="mb-2 block text-xs font-medium select-none" for="server-name-{id}">
			Display name
		</label>

		<Input
			id="server-name-{id}"
			oninput={(e) => onNameChange?.(e.currentTarget.value)}
			placeholder={namePlaceholder}
			type="text"
			value={name}
		/>
	</div>

	<label class="flex items-center gap-2 cursor-pointer select-none">
		<Switch
			checked={showAuthorization}
			disabled={required}
			id="use-authorization-{id}"
			onCheckedChange={setUseAuthorization}
		/>

		<span class="text-xs text-muted-foreground">
			Authorization{#if required}
				<span class="text-destructive">*</span>{/if}
		</span>
	</label>

	{#if showAuthorization}
		<div class="relative mt-2">
			<Input
				bind:ref={bearerInput}
				autocomplete="off"
				class="pl-16"
				id="bearer-token-{id}"
				oninput={(e) => updateBearerToken(e.currentTarget.value)}
				placeholder="Paste token here"
				type="password"
				value={bearerToken}
			/>

			<span
				class="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-foreground"
			>
				Bearer
			</span>
		</div>
	{/if}

	<KeyValuePairs
		addButtonLabel="Add"
		class="mt-3"
		emptyMessage="No custom headers configured."
		keyPlaceholder="Header name"
		onPairsChange={(pairs) => {
			const auth = headerPairs.find(ownedByBearerUi);

			updateHeaderPairs(auth ? [...pairs, auth] : pairs);
		}}
		pairs={headerPairs.filter((p) => !ownedByBearerUi(p))}
		sectionLabel="Custom Headers"
		sectionLabelOptional
		valuePlaceholder="Value"
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
				checked={useProxy}
				class="mt-1"
				disabled={!mcpStore.isProxyAvailable}
				id="use-proxy-{id}"
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
