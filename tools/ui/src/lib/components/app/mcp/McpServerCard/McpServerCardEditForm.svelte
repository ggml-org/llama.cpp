<script lang="ts">
	import { McpServerForm } from '$lib/components/app/mcp';
	import { Button } from '$lib/components/ui/button';
	import { parseHeadersToArray } from '$lib/utils';

	interface Props {
		serverId: string;
		serverUrl: string;
		serverUseProxy?: boolean;
		/** Current automatic label, prefilled so the user can customize it. */
		serverLabel?: string;
		onSave: (url: string, headers: string, useProxy: boolean, name?: string) => void;
		onCancel: () => void;
	}

	let {
		onCancel,
		onSave,
		serverId,
		serverLabel = '',
		serverUrl,
		serverUseProxy = false
	}: Props = $props();

	let editUrl = $derived(serverUrl);
	let editName = $derived(serverLabel);
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

		// An unchanged prefill keeps following the automatic label; only an
		// actual edit becomes a persisted custom display name.
		const name = editName.trim() !== serverLabel.trim() ? editName.trim() : undefined;

		onSave(editUrl.trim(), editHeaders.trim(), editUseProxy, name);
	}

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		handleSave();
	}

	export function setInitialValues(url: string, headers: string, useProxy: boolean, name = '') {
		editUrl = url;
		editHeaders = headers;
		editUseProxy = useProxy;
		editName = name;
	}
</script>

<form class="contents" onsubmit={handleSubmit}>
	<div class="space-y-4">
		<p class="font-medium">Configure Server</p>

		<McpServerForm
			headers={editHeaders}
			id={serverId}
			name={editName}
			onHeadersChange={(v) => (editHeaders = v)}
			onNameChange={(v) => (editName = v)}
			onUrlChange={(v) => (editUrl = v)}
			onUseProxyChange={(v) => (editUseProxy = v)}
			url={editUrl}
			urlError={editUrl ? urlError : null}
			useProxy={editUseProxy}
		/>

		<div class="flex items-center justify-end gap-2">
			<Button onclick={onCancel} size="sm" variant="secondary">Cancel</Button>

			<Button disabled={!canSave} size="sm" type="submit">
				{serverUrl.trim() ? 'Update' : 'Add'}
			</Button>
		</div>
	</div>
</form>
