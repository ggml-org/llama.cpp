<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { uuid } from '$lib/utils';
	import { MCP_SERVER_ID_PREFIX } from '$lib/constants';

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

	function handleOpenChange(value: boolean) {
		if (!value) {
			newServerUrl = '';
			newServerHeaders = '';
		}
		open = value;
		onOpenChange?.(value);
	}

	function saveNewServer() {
		if (newServerUrlError) return;

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
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-md">
		<Dialog.Header>
			<Dialog.Title>Add New Server</Dialog.Title>
		</Dialog.Header>

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
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button
				variant="default"
				size="sm"
				onclick={saveNewServer}
				disabled={!!newServerUrlError}
				aria-label="Save"
			>
				Add
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
