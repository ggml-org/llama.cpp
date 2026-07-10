<script lang="ts">
	import { Sparkles } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Empty from '$lib/components/ui/empty';
	import { McpServerCardCompact, McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { parseHeadersToArray, uuid } from '$lib/utils';
	import {
		DISMISSED_RECOMMENDED_MCP_SERVER_IDS_LOCALSTORAGE_KEY,
		DEFAULT_MCP_CONFIG,
		MCP_RECOMMENDATIONS_CONSENT_LOCALSTORAGE_KEY,
		MCP_SERVER_ID_PREFIX,
		RECOMMENDED_MCP_SERVERS
	} from '$lib/constants';
	import { browser } from '$app/environment';

	type ConsentState = 'undecided' | 'agreed' | 'dismissed';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let newServerUrl = $state('');
	let newServerHeaders = $state('');
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
	let canSave = $derived(!newServerUrlError && newServerHeaderPairsValid);

	let previewServer = $derived.by(() => {
		if (!newServerUrl.trim() || newServerUrlError) return null;

		return {
			id: `preview:${newServerUrl.trim()}`,
			name: undefined,
			url: newServerUrl.trim(),
			enabled: true,
			requestTimeoutSeconds: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
			headers: newServerHeaders.trim() || undefined
		};
	});

	function readConsent(): ConsentState {
		if (!browser) return 'undecided';
		const raw = localStorage.getItem(MCP_RECOMMENDATIONS_CONSENT_LOCALSTORAGE_KEY);

		// 'true' / 'agreed' maps to consented; everything else (including
		// legacy 'dismissed' / 'false') maps to dismissed. Missing value is
		// treated as undecided so the prompt still appears.
		if (raw === 'true' || raw === 'agreed') return 'agreed';
		if (raw === 'false' || raw === 'dismissed') return 'dismissed';
		return 'undecided';
	}

	function writeConsent(value: 'agreed' | 'dismissed') {
		consent = value;

		if (browser) {
			// Persist as a boolean literal so the value matches the
			// `mcpRecommendationsConsent` key's intent.
			localStorage.setItem(
				MCP_RECOMMENDATIONS_CONSENT_LOCALSTORAGE_KEY,
				value === 'agreed' ? 'true' : 'false'
			);
		}
	}

	function readDismissedRecommendationIds(): string[] {
		if (!browser) return [];
		const raw = localStorage.getItem(DISMISSED_RECOMMENDED_MCP_SERVER_IDS_LOCALSTORAGE_KEY);

		if (!raw) return [];

		try {
			const parsed = JSON.parse(raw);
			return Array.isArray(parsed)
				? parsed.filter((id): id is string => typeof id === 'string')
				: [];
		} catch {
			return [];
		}
	}

	function writeDismissedRecommendationIds(ids: string[]) {
		dismissedRecommendationIds = ids;

		if (browser) {
			localStorage.setItem(
				DISMISSED_RECOMMENDED_MCP_SERVER_IDS_LOCALSTORAGE_KEY,
				JSON.stringify(ids)
			);
		}
	}

	let consent = $state<ConsentState>(readConsent());
	let dismissedRecommendationIds = $state<string[]>(readDismissedRecommendationIds());

	// A recommendation is "selected" iff the form URL exactly matches its
	// URL. Clicking a card writes its URL into the form, which lights it
	// up; editing the URL away from any recommendation's URL clears the
	// selection automatically - no separate state to keep in sync.
	let selectedRecommendationId = $derived.by(() => {
		const url = newServerUrl.trim();
		if (!url) return null;
		return RECOMMENDED_MCP_SERVERS.find((rec) => rec.url === url)?.id ?? null;
	});
	let hasSelection = $derived(selectedRecommendationId !== null);

	// Cross-check against the configured MCP servers by URL — a recommended
	// entry whose URL is already in the config is filtered out of the slot.
	let unconfiguredRecommendations = $derived(
		RECOMMENDED_MCP_SERVERS.filter(
			(rec) => !mcpStore.getServers().some((configured) => configured.url === rec.url)
		)
	);

	let recommendationsToShow = $derived(
		consent === 'agreed'
			? unconfiguredRecommendations.filter((rec) => !dismissedRecommendationIds.includes(rec.id))
			: []
	);

	function handleAgree() {
		writeConsent('agreed');
	}

	function handleDismissAll() {
		writeConsent('dismissed');
	}

	function handleRecommendationClick(recommendedId: string) {
		const recommendation = RECOMMENDED_MCP_SERVERS.find((rec) => rec.id === recommendedId);

		if (!recommendation) return;

		// Fill the form so the user can review / tweak before saving through
		// the dialog's primary "Add" button. The selection highlight follows
		// from the URL match (see selectedRecommendationId).
		newServerUrl = recommendation.url;
		newServerHeaders = '';
	}

	function handleDismissRecommendation(recommendedId: string) {
		if (dismissedRecommendationIds.includes(recommendedId)) return;

		writeDismissedRecommendationIds([...dismissedRecommendationIds, recommendedId]);
	}

	function handleOpenChange(value: boolean) {
		if (!value) {
			if (previewServer) {
				mcpStore.clearHealthCheck(previewServer.id);
			}
			newServerUrl = '';
			newServerHeaders = '';
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
			headers: newServerHeaders.trim() || undefined
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
	<Dialog.Content class="sm:max-w-3xl" onOpenAutoFocus={(event) => event.preventDefault()}>
		<Dialog.Header>
			<Dialog.Title>Add New Server</Dialog.Title>
		</Dialog.Header>

		{#if consent === 'undecided'}
			<Empty.Root class="border mt-4">
				<Empty.Header>
					<Empty.Media variant="icon">
						<Sparkles />
					</Empty.Media>
					<Empty.Title>Show recommended MCP servers</Empty.Title>
					<Empty.Description>Briefly connects to exa.ai and huggingface.co.</Empty.Description>
				</Empty.Header>
				<Empty.Content>
					<div class="flex justify-center gap-2">
						<Button size="sm" variant="secondary" onclick={handleDismissAll}>Not now</Button>

						<Button size="sm" onclick={handleAgree} aria-label="Load recommended MCP servers">
							Load recommendations
						</Button>
					</div>
				</Empty.Content>
			</Empty.Root>
		{:else if recommendationsToShow.length > 0}
			<div class="space-y-3 pt-2">
				<h3 class="text-sm font-medium text-muted-foreground">Recommended</h3>

				<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
					{#each recommendationsToShow as recommendation (recommendation.id)}
						<McpServerCardCompact
							server={{
								id: recommendation.id,
								name: recommendation.name,
								url: recommendation.url,
								description: recommendation.description,
								enabled: true,
								requestTimeoutSeconds: recommendation.requestTimeoutSeconds
							}}
							onClick={() => handleRecommendationClick(recommendation.id)}
							onDismiss={() => handleDismissRecommendation(recommendation.id)}
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
					onUrlChange={(v) => (newServerUrl = v)}
					onHeadersChange={(v) => (newServerHeaders = v)}
					urlError={newServerUrl ? newServerUrlError : null}
					id="new-server"
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
