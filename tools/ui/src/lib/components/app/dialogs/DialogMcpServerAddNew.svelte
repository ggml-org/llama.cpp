<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Loader2 } from '@lucide/svelte';
	import { McpServerForm } from '$lib/components/app/mcp';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { uuid } from '$lib/utils';
	import { DEFAULT_MCP_CONFIG, MCP_SERVER_ID_PREFIX } from '$lib/constants';
	import { HealthCheckStatus } from '$lib/enums';

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
	let isDuplicateUrl = $derived(
		newServerUrl.trim().length > 0 && mcpStore.hasServerWithUrl(newServerUrl.trim())
	);
	let isSaving = $state(false);
	let connectionError = $state<string | null>(null);
	let urlFieldError = $derived(
		connectionError ??
			(newServerUrl
				? isDuplicateUrl
					? 'A server with this URL already exists'
					: newServerUrlError
				: null)
	);

	function handleOpenChange(value: boolean) {
		if (!value) {
			newServerUrl = '';
			newServerHeaders = '';
			connectionError = null;
		}
		open = value;
		onOpenChange?.(value);
	}

	async function saveNewServer() {
		if (newServerUrlError || isDuplicateUrl || isSaving) return;

		const newServerId = uuid() ?? `${MCP_SERVER_ID_PREFIX}-${Date.now()}`;
		const trimmedUrl = newServerUrl.trim();
		const trimmedHeaders = newServerHeaders.trim();
		const requestTimeoutSeconds =
			Number(config().mcpRequestTimeoutSeconds) || DEFAULT_MCP_CONFIG.requestTimeoutSeconds;

		isSaving = true;
		connectionError = null;

		try {
			// Validate the URL by attempting to connect. Promotes to active
			// connection on success so the saved server is immediately usable.
			await mcpStore.runHealthCheck(
				{
					id: newServerId,
					enabled: true,
					url: trimmedUrl,
					requestTimeoutSeconds,
					headers: trimmedHeaders || undefined
				},
				true
			);

			const state = mcpStore.getHealthCheckState(newServerId);

			if (state.status !== HealthCheckStatus.SUCCESS) {
				mcpStore.clearHealthCheck(newServerId);
				connectionError =
					state.status === HealthCheckStatus.ERROR
						? state.message || 'Could not connect to the server.'
						: 'Could not connect to the server.';
				return;
			}

			const added = mcpStore.addServer({
				id: newServerId,
				enabled: true,
				url: trimmedUrl,
				headers: trimmedHeaders || undefined
			});

			if (!added) {
				mcpStore.clearHealthCheck(newServerId);
				return;
			}

			conversationsStore.setMcpServerOverride(newServerId, true);

			handleOpenChange(false);
		} finally {
			isSaving = false;
		}
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
				onUrlChange={(v) => {
					newServerUrl = v;
					connectionError = null;
				}}
				onHeadersChange={(v) => (newServerHeaders = v)}
				urlError={urlFieldError}
				id="new-server"
			/>
		</div>

		<Dialog.Footer>
			<Button
				variant="secondary"
				size="sm"
				onclick={() => handleOpenChange(false)}
				disabled={isSaving}
			>
				Cancel
			</Button>

			<Button
				variant="default"
				size="sm"
				onclick={saveNewServer}
				disabled={!!newServerUrlError || isDuplicateUrl || isSaving}
				aria-label="Save"
			>
				{#if isSaving}
					<Loader2 class="h-4 w-4 animate-spin" />

					Adding...
				{:else}
					Add
				{/if}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
