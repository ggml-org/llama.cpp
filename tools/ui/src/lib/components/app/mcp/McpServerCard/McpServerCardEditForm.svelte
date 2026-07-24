<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { McpServerForm } from '$lib/components/app/mcp';
	import { parseHeadersToArray } from '$lib/utils';

	interface Props {
		serverId: string;
		serverUrl: string;
		serverUseProxy?: boolean;
		onSave: (url: string, headers: string, useProxy: boolean) => void;
		onCancel: () => void;
	}

	let { serverId, serverUrl, serverUseProxy = false, onSave, onCancel }: Props = $props();

	let editUrl = $derived(serverUrl);
	let editHeaders = $state('');
	let editUseProxy = $derived(serverUseProxy);

	let urlError = $derived.by(() => {
		if (!editUrl.trim()) return 'URL is required';
		try {
			new URL(editUrl);
			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	let headerPairsValid = $derived(
		parseHeadersToArray(editHeaders).every((p) => p.key.trim() && p.value.trim())
	);
	let canSave = $derived(!urlError && headerPairsValid);

	function handleSave() {
		if (!canSave) return;
		onSave(editUrl.trim(), editHeaders.trim(), editUseProxy);
	}

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		handleSave();
	}

	export function setInitialValues(url: string, headers: string, useProxy: boolean) {
		editUrl = url;
		editHeaders = headers;
		editUseProxy = useProxy;
	}
</script>

<form onsubmit={handleSubmit} class="contents">
	<div class="space-y-4">
		<p class="font-medium">Configure Server</p>

		<McpServerForm
			url={editUrl}
			headers={editHeaders}
			useProxy={editUseProxy}
			onUrlChange={(v) => (editUrl = v)}
			onHeadersChange={(v) => (editHeaders = v)}
			onUseProxyChange={(v) => (editUseProxy = v)}
			urlError={editUrl ? urlError : null}
			id={serverId}
		/>

		<div class="flex items-center justify-end gap-2">
			<Button variant="secondary" size="sm" onclick={onCancel}>Cancel</Button>

			<Button size="sm" type="submit" disabled={!canSave}>
				{serverUrl.trim() ? 'Update' : 'Add'}
			</Button>
		</div>
	</div>
</form>
