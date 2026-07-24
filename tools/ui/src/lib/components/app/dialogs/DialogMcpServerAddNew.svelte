<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { McpServerCardCompact, McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { parseHeadersToArray, uuid, canonicalizeServerUrl } from '$lib/utils';
	import {
		BEARER_PREFIX,
		BOOL_FALSE_STRING,
		BOOL_TRUE_STRING,
		DISMISSED_RECOMMENDED_MCP_SERVERS_LOCALSTORAGE_KEY,
		MCP_SERVER_ID_PREFIX,
		RECOMMENDED_MCP_SERVERS,
		REDACTED_HEADERS
	} from '$lib/constants';
	import { browser } from '$app/environment';
	import { HealthCheckStatus } from '$lib/enums';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let newServerUrl = $state('');
	let newServerName = $state('');
	let nameAutoFilled = $state('');
	let nameTouched = $state(false);

	let previewRun = 0;

	function handleNameChange(value: string) {
		newServerName = value;
		nameTouched = true;
	}
	let newServerHeaders = $state('');
	let newServerUseProxy = $state(false);

	let newServerWantsAuthorization = $state(false);

	let selectedRecommendationId = $derived.by(() => {
		const url = newServerUrl.trim();
		if (!url) return null;
		const targetCanonical = canonicalizeServerUrl(url);
		return (
			RECOMMENDED_MCP_SERVERS.find((rec) => canonicalizeServerUrl(rec.url) === targetCanonical)
				?.id ?? null
		);
	});
	let selectedRecommendation = $derived(
		selectedRecommendationId
			? (RECOMMENDED_MCP_SERVERS.find((rec) => rec.id === selectedRecommendationId) ?? null)
			: null
	);
	let authRequired = $derived(selectedRecommendation?.needsAuthorization ?? false);

	let bearerTokenFilled = $derived.by(() => {
		const pairs = parseHeadersToArray(newServerHeaders);
		const bearerPrefix = BEARER_PREFIX.toLowerCase();
		const bearer = pairs.find(
			(p) =>
				REDACTED_HEADERS.has(p.key.trim().toLowerCase()) &&
				p.value.trim().toLowerCase().startsWith(bearerPrefix)
		);

		if (!bearer) return false;

		return bearer.value.trim().slice(bearerPrefix.length).trim().length > 0;
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

	// Backward-compatible read: older versions stored a JSON array of dismissed ids.
	function readRecommendationsDismissed(): boolean {
		if (!browser) return false;
		const raw = localStorage.getItem(DISMISSED_RECOMMENDED_MCP_SERVERS_LOCALSTORAGE_KEY);

		if (!raw) return false;

		if (raw === BOOL_TRUE_STRING) return true;
		if (raw === BOOL_FALSE_STRING) return false;

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
				dismissed ? BOOL_TRUE_STRING : BOOL_FALSE_STRING
			);
		}
	}

	let recommendationsDismissed = $state<boolean>(readRecommendationsDismissed());

	// Read-only once a recommendation is picked: switch is disabled, so we keep
	// the Authorization field in sync with the requirement.
	$effect(() => {
		if (authRequired) {
			newServerWantsAuthorization = true;
		}
	});

	// Debounced preview handshake: once the URL is valid and stable, fetch the
	// server-reported name to prefill the display name field. A manual edit
	// freezes the autofill for good, and failures stay silent.
	$effect(() => {
		const url = newServerUrl.trim();
		const headers = newServerHeaders.trim();
		const useProxy = newServerUseProxy;

		if (!open || newServerUrlError || !url) return;

		const run = ++previewRun;
		// One throwaway id per run: concurrent previews (URL typed, then the
		// bearer token pasted) would poison each other's shared health state.
		const previewId = `${MCP_SERVER_ID_PREFIX}-preview-${run}`;
		const timer = setTimeout(async () => {
			await mcpStore.runHealthCheck({
				id: previewId,
				enabled: false,
				url,
				headers: headers || undefined,
				useProxy
			});

			const state = mcpStore.getHealthCheckState(previewId);

			mcpStore.clearHealthCheck(previewId);

			if (run !== previewRun) return;

			if (state.status !== HealthCheckStatus.SUCCESS) return;

			const autoName = state.serverInfo?.title || state.serverInfo?.name || '';

			if (autoName && !nameTouched) {
				newServerName = autoName;
				nameAutoFilled = autoName;
			}
		}, 600);

		return () => clearTimeout(timer);
	});

	let hasSelection = $derived(selectedRecommendationId !== null);

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

		newServerUrl = recommendation.url;
		newServerHeaders = '';
		newServerWantsAuthorization = recommendation.needsAuthorization ?? false;
	}

	function handleDismissAll() {
		writeRecommendationsDismissed(true);
	}

	function handleOpenChange(value: boolean) {
		if (!value) {
			newServerUrl = '';
			newServerName = '';
			nameAutoFilled = '';
			nameTouched = false;
			previewRun++;
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
			// A name equal to the autofilled server-reported one is not a
			// customization: keep following the automatic label.
			displayName:
				newServerName.trim() && newServerName.trim() !== nameAutoFilled.trim()
					? newServerName.trim()
					: undefined,
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
					<h3 class="text-sm font-medium">Recommended Servers</h3>
					<Button class="text-muted-foreground" variant="ghost" size="sm" onclick={handleDismissAll}
						>Dismiss</Button
					>
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
					name={newServerName}
					onNameChange={handleNameChange}
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
