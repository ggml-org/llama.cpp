<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { McpServerForm } from '$lib/components/app/mcp';

	interface Props {
		serverId: string;
		serverUrl: string;
		serverUseProxy?: boolean;
		serverOauth?: boolean;
		onSave: (url: string, headers: string, useProxy: boolean, oauth: boolean) => void;
		onCancel: () => void;
	}

	let { serverId, serverUrl, serverUseProxy = false, serverOauth = false, onSave, onCancel }: Props =
		$props();

	let editUrl = $derived(serverUrl);
	let editHeaders = $state('');
	let editUseProxy = $derived(serverUseProxy);
	let editOauth = $derived(serverOauth);

	let urlError = $derived.by(() => {
		if (!editUrl.trim()) return 'URL is required';
		try {
			new URL(editUrl);
			return null;
		} catch {
			return 'Invalid URL format';
		}
	});

	let canSave = $derived(!urlError);

	function handleSave() {
		if (!canSave) return;
		onSave(editUrl.trim(), editHeaders.trim(), editUseProxy, editOauth);
	}

	export function setInitialValues(url: string, headers: string, useProxy: boolean, oauth: boolean) {
		editUrl = url;
		editHeaders = headers;
		editUseProxy = useProxy;
		editOauth = oauth;
	}
</script>

<div class="space-y-4">
	<p class="font-medium">Configure Server</p>

	<McpServerForm
		url={editUrl}
		headers={editHeaders}
		useProxy={editUseProxy}
		oauth={editOauth}
		onUrlChange={(v) => (editUrl = v)}
		onHeadersChange={(v) => (editHeaders = v)}
		onUseProxyChange={(v) => (editUseProxy = v)}
		onOAuthChange={(v) => (editOauth = v)}
		urlError={editUrl ? urlError : null}
		id={serverId}
	/>

	<div class="flex items-center justify-end gap-2">
		<Button variant="secondary" size="sm" onclick={onCancel}>Cancel</Button>

		<Button size="sm" onclick={handleSave} disabled={!canSave}>
			{serverUrl.trim() ? 'Update' : 'Add'}
		</Button>
	</div>
</div>
