<script lang="ts">
	import { X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import McpServerForm from '$lib/components/app/mcp/McpServerForm.svelte';

	interface Props {
		serverId: string;
		serverUrl: string;
		onSave: (url: string, headers: string) => void;
		onCancel: () => void;
	}

	let { serverId, serverUrl, onSave, onCancel }: Props = $props();

	let editUrl = $state(serverUrl);
	let editHeaders = $state('');

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
		onSave(editUrl.trim(), editHeaders.trim());
	}

	export function setInitialValues(url: string, headers: string) {
		editUrl = url;
		editHeaders = headers;
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<p class="font-medium">Configure Server</p>
		<Button variant="ghost" size="icon" class="h-7 w-7" onclick={onCancel} aria-label="Cancel">
			<X class="h-3.5 w-3.5" />
		</Button>
	</div>

	<McpServerForm
		url={editUrl}
		headers={editHeaders}
		onUrlChange={(v) => (editUrl = v)}
		onHeadersChange={(v) => (editHeaders = v)}
		urlError={editUrl ? urlError : null}
		id={serverId}
	/>

	<div class="flex items-center justify-end">
		<Button variant="default" size="sm" onclick={handleSave} disabled={!canSave} aria-label="Save">
			{serverUrl.trim() ? 'Update' : 'Add'}
		</Button>
	</div>
</div>
