import { browser } from '$app/environment';
import {
	MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY,
	RECOMMENDED_MCP_SERVER_IDS,
	RECOMMENDED_MCP_SERVERS_OPTIN_DIALOG_DELAY
} from '$lib/constants';
import { mcpStore } from '$lib/stores/mcp.svelte';

/**
 * First-run opt-in dialog for the recommended MCP servers.
 *
 * Owns the dismissed / open / trigger-timeout state and the effect that
 * schedules the dialog. Reads opt-in status and the configured server list
 * from `mcpStore`, so callers don't need to recompute on their side.
 */
export function useMcpRecommendations() {
	let dismissed = $state(
		browser && localStorage.getItem(MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY) === 'true'
	);
	let open = $state(false);
	let checked = $state(false);
	let triggerTimeout: ReturnType<typeof setTimeout> | null = null;

	function dismiss() {
		if (browser) {
			localStorage.setItem(MCP_SERVERS_ADDED_TO_CHAT_LOCALSTORAGE_KEY, 'true');
		}
		dismissed = true;
		open = false;
		if (triggerTimeout) {
			clearTimeout(triggerTimeout);
			triggerTimeout = null;
		}
	}

	function handleOpenChange(next: boolean) {
		open = next;
		if (!next) dismiss();
	}

	$effect(() => {
		if (!browser) return;

		if (open || dismissed) {
			if (triggerTimeout) {
				clearTimeout(triggerTimeout);
				triggerTimeout = null;
			}
			return;
		}

		// Already evaluated once this session; leave any pending trigger alone so
		// it can still fire later. Setting `checked = true` below re-runs this
		// effect, and we must not wipe the timeout that was just scheduled.
		if (checked) return;

		const hasRecommendations = mcpStore
			.getServers()
			.some((server) => RECOMMENDED_MCP_SERVER_IDS.has(server.id));

		if (hasRecommendations) {
			triggerTimeout = setTimeout(() => {
				open = true;
			}, RECOMMENDED_MCP_SERVERS_OPTIN_DIALOG_DELAY);
		}

		checked = true;
	});

	return {
		get open() {
			return open;
		},
		get dismissed() {
			return dismissed;
		},
		dismiss,
		handleOpenChange
	};
}
