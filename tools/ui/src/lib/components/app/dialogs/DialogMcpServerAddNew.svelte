<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { McpServerCardCompact, McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { parseHeadersToArray, uuid, canonicalizeServerUrl } from '$lib/utils';
	import {
		DISMISSED_RECOMMENDED_MCP_SERVERS_LOCALSTORAGE_KEY,
		MCP_SERVER_ID_PREFIX,
		RECOMMENDED_MCP_SERVERS
	} from '$lib/constants';
	import { browser } from '$app/environment';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let newServerUrl = $state('');
	let newServerHeaders = $state('');
	let newServerUseProxy = $state(false);

	// Mirrored from the form via `bind:` so the recommendation click handler
	// can flip the "Authorization" switch on for servers whose
	// `needsAuthorization` flag is true.
	let newServerWantsAuthorization = $state(false);

	// A recommendation is "selected" iff the form URL exactly matches its
	// URL. Clicking a card writes its URL into the form, which lights it
	// up; editing the URL away from any recommendation's URL clears the
	// selection automatically - no separate state to keep in sync.
	let selectedRecommendationId = $derived.by(() => {
		const url = newServerUrl.trim();
		if (!url) return null;
		return RECOMMENDED_MCP_SERVERS.find((rec) => rec.url === url)?.id ?? null;
	});
	let selectedRecommendation = $derived(
		selectedRecommendationId
			? (RECOMMENDED_MCP_SERVERS.find((rec) => rec.id === selectedRecommendationId) ?? null)
			: null
	);
	let authRequired = $derived(selectedRecommendation?.needsAuthorization ?? false);

	// Reflects whether the Authorization: Bearer header currently carries a
	// non-empty token. Auth-required recommendations gate the Add button on
	// this rather than on `hasAuthorization` (which would also light up for
	// empty "Bearer " entries).
	let bearerTokenFilled = $derived.by(() => {
		const pairs = parseHeadersToArray(newServerHeaders);
		const bearer = pairs.find(
			(p) =>
				p.key.trim().toLowerCase() === 'authorization' &&
				p.value.trim().toLowerCase().startsWith('bearer ')
		);

		if (!bearer) return false;

		return bearer.value.trim().slice('bearer '.length).trim().length > 0;
	});

	let newServerUrlError = $derived.by(() => {
		if (!newServerUrl.trim()) return 'URL is required';
		try {
			new URL(newServerUrl);

			return null;
		} catch {
			return 'Invalid URL format';
		}
	});
	let newServerHeaderPairsValid = $derived(
		parseHeadersToArray(newServerHeaders).every((p) => p.key.trim() && p.value.trim())
	);
	let canSave = $derived(
		!newServerUrlError && newServerHeaderPairsValid && (!authRequired || bearerTokenFilled)
	);

	// Stored as a boolean string ("true"/"false"). Reads also tolerate the
	// legacy JSON-array payload: anything non-empty in the old shape counted
	// as "dismissed" because the only writer back then wrote every rec id at
	// once, so users who dismissed under the previous schema keep the
	// section hidden after upgrading.
	function readRecommendationsDismissed(): boolean {
		if (!browser) return false;
		const raw = localStorage.getItem(DISMISSED_RECOMMENDED_MCP_SERVERS_LOCALSTORAGE_KEY);

		if (!raw) return false;

		if (raw === 'true') return true;
		if (raw === 'false') return false;

		try {
			const parsed = JSON.parse(raw);
			return Array.isArray(parsed) && parsed.length > 0;
		} catch {
			return false;
		}
	}

	function writeRecommendationsDismissed(dismissed: boolean) {
		recommendationsDismissed = dismissed;

		if (browser) {
			localStorage.setItem(
				DISMISSED_RECOMMENDED_MCP_SERVERS_LOCALSTORAGE_KEY,
				dismissed ? 'true' : 'false'
			);
		}
	}

	let recommendationsDismissed = $state<boolean>(readRecommendationsDismissed());

	// Keep the Authorization intent on whenever the picked recommendation
	// requires it. Pairs with the `disabled` switch on the form so the user
	// can't end up on a needed-auth URL with the toggle off and no way to
	// recover - they'd have to either retype the URL or delete the auth
	// entry from the KV grid manually.
	$effect(() => {
		if (authRequired) {
			newServerWantsAuthorization = true;
		}
	});

	let hasSelection = $derived(selectedRecommendationId !== null);

	// Cross-check against the configured MCP servers by URL - a recommended
	// entry whose URL is already in the config is filtered out of the slot.
	// Dismissed entries are also filtered out, so the section disappears
	// entirely once the user has dismissed everything. URLs are normalized
	// before comparing so a stored `https://api.example.com/mcp/` doesn't
	// slip past the recommended `https://api.example.com/mcp`.
	let unconfiguredRecommendations = $derived.by(() => {
		const configuredCanonicals = new Set(
			mcpStore.getServers().map((s) => canonicalizeServerUrl(s.url))
		);

		return RECOMMENDED_MCP_SERVERS.filter(
			(rec) => !configuredCanonicals.has(canonicalizeServerUrl(rec.url))
		);
	});

	let recommendationsToShow = $derived(recommendationsDismissed ? [] : unconfiguredRecommendations);

	function handleRecommendationClick(recommendedId: string) {
		const recommendation = RECOMMENDED_MCP_SERVERS.find((rec) => rec.id === recommendedId);

		if (!recommendation) return;

		// Fill the form so the user can review / tweak before saving through
		// the dialog's primary "Add" button. The selection highlight follows
		// from the URL match (see selectedRecommendationId).
		newServerUrl = recommendation.url;
		newServerHeaders = '';
		// Honor the server's auth requirement - servers flagged
		// `needsAuthorization: true` flip the form's "Authorization" switch on
		// so the user can paste a Bearer token right away. Servers without
		// the flag turn the switch off, so pre-existing tokens don't leak
		// across recommendation picks.
		newServerWantsAuthorization = recommendation.needsAuthorization ?? false;
	}

	function handleDismissAll() {
		// Keep the section hidden on future opens, even after the user
		// re-configures a server that currently makes one disappear from
		// the unconfigured list above.
		writeRecommendationsDismissed(true);
	}

	function handleOpenChange(value: boolean) {
		if (!value) {
			newServerUrl = '';
			newServerHeaders = '';
			newServerUseProxy = false;
			newServerWantsAuthorization = false;
		}
		open = value;
		onOpenChange?.(value);
	}

	function saveNewServer() {
		if (!canSave) return;

		const newServerId = uuid() ?? `${MCP_SERVER_ID_PREFIX}-${Date.now()}`;

		mcpStore.addServer({
			id: newServerId,
			enabled: true,
			url: newServerUrl.trim(),
			headers: newServerHeaders.trim() || undefined,
			useProxy: newServerUseProxy
		});

		conversationsStore.setMcpServerOverride(newServerId, true);

		handleOpenChange(false);
	}

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		saveNewServer();
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-2xl">
		<Dialog.Header>
			<Dialog.Title class="select-none">Add New MCP Server</Dialog.Title>
		</Dialog.Header>

		{#if recommendationsToShow.length > 0}
			<div class="space-y-3 pt-2">
				<div class="flex items-center justify-between gap-3">
					<h3 class="text-sm font-medium text-muted-foreground">Recommended Servers</h3>
					<Button variant="ghost" size="sm" onclick={handleDismissAll}>Dismiss</Button>
				</div>

				<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
					{#each recommendationsToShow as recommendation (recommendation.id)}
						<McpServerCardCompact
							server={recommendation}
							onClick={() => handleRecommendationClick(recommendation.id)}
							selected={selectedRecommendationId === recommendation.id}
							dimmed={hasSelection && selectedRecommendationId !== recommendation.id}
						/>
					{/each}
				</div>
			</div>
		{/if}

		<form onsubmit={handleSubmit} class="contents">
			<div class="space-y-4 py-4">
				<McpServerForm
					url={newServerUrl}
					headers={newServerHeaders}
					useProxy={newServerUseProxy}
					onUrlChange={(v) => (newServerUrl = v)}
					onHeadersChange={(v) => (newServerHeaders = v)}
					onUseProxyChange={(v) => (newServerUseProxy = v)}
					urlError={newServerUrl ? newServerUrlError : null}
					id="new-server"
					bind:wantsAuthorization={newServerWantsAuthorization}
					required={authRequired}
				/>
			</div>

			<Dialog.Footer>
				<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>
					Cancel
				</Button>

				<Button variant="default" size="sm" type="submit" disabled={!canSave} aria-label="Save">
					Add
				</Button>
			</Dialog.Footer>
		</form>
	</Dialog.Content>
</Dialog.Root>
