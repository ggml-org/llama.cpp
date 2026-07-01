<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';

	interface Props {
		serverId: string;
		serverUrl: string;
		serverUseProxy?: boolean;
		onSave: (url: string, headers: string, useProxy: boolean) => boolean;
		onCancel: () => void;
	}

	let { serverId, serverUrl, serverUseProxy = false, onSave, onCancel }: Props = $props();

	let editUrl = $derived(serverUrl);
	let editHeaders = $state('');
	let editUseProxy = $derived(serverUseProxy);

	let urlError = $derived.by(() => {
		const trimmed = editUrl.trim();
		if (!trimmed) return 'URL is required';
		try {
			new URL(trimmed);
			return null;
		} catch {
			return 'Invalid URL format';
		}
	});
	let isDuplicateUrl = $derived.by(() => {
		const trimmed = editUrl.trim();
		if (!trimmed) return false;
		const matches = mcpStore
			.getServers()
			.filter((s) => s.id !== serverId && s.url.trim() === trimmed);
		return matches.length > 0;
	});

	let canSave = $derived(!urlError && !isDuplicateUrl);
	let displayUrlError = $derived(
		editUrl ? (isDuplicateUrl ? 'A server with this URL already exists' : urlError) : null
	);

	function handleSave() {
		if (!canSave) return;
		const saved = onSave(editUrl.trim(), editHeaders.trim(), editUseProxy);
		if (!saved) {
			return;
		}
	}

	export function setInitialValues(url: string, headers: string, useProxy: boolean) {
		editUrl = url;
		editHeaders = headers;
		editUseProxy = useProxy;
	}
</script>

<div class="space-y-4">
	<p class="font-medium">Configure Server</p>

	<McpServerForm
		url={editUrl}
		headers={editHeaders}
		useProxy={editUseProxy}
		onUrlChange={(v) => (editUrl = v)}
		onHeadersChange={(v) => (editHeaders = v)}
		onUseProxyChange={(v) => (editUseProxy = v)}
		urlError={displayUrlError}
		id={serverId}
	/>

	<div class="flex items-center justify-end gap-2">
		<Button variant="secondary" size="sm" onclick={onCancel}>Cancel</Button>

		<Button size="sm" onclick={handleSave} disabled={!canSave}>
			{serverUrl.trim() ? 'Update' : 'Add'}
		</Button>
	</div>
</div>
