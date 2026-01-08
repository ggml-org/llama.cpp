<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import McpSettingsSection from '../mcp/McpSettingsSection.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { Button } from '$lib/components/ui/button';
	import McpLogo from '../misc/McpLogo.svelte';

	interface Props {
		onOpenChange?: (open: boolean) => void;
		open?: boolean;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	let localConfig = $state(config());

	function handleClose() {
		onOpenChange?.(false);
	}

	function handleConfigChange(key: string, value: string | boolean) {
		localConfig = { ...localConfig, [key]: value };
	}

	function handleSave() {
		// Save all changes to settingsStore
		Object.entries(localConfig).forEach(([key, value]) => {
			if (config()[key as keyof typeof localConfig] !== value) {
				settingsStore.updateConfig(key as keyof typeof localConfig, value);
			}
		});
		onOpenChange?.(false);
	}

	function handleCancel() {
		// Reset to current config
		localConfig = config();
		onOpenChange?.(false);
	}

	$effect(() => {
		if (open) {
			// Reset local config when dialog opens
			localConfig = config();
		}
	});
</script>

<Dialog.Root {open} onOpenChange={handleClose}>
	<Dialog.Content
		class="z-999999 flex h-[100dvh] max-h-[100dvh] min-h-[100dvh] flex-col gap-0 rounded-none p-0
			md:h-[80vh] md:max-h-[80vh] md:min-h-0 md:rounded-lg"
		style="max-width: 56rem;"
	>
		<div class="border-b p-4 md:p-6">
			<Dialog.Title class="inline-flex items-center text-lg font-semibold">
				<McpLogo class="mr-2 inline h-4 w-4" />

				MCP Servers
			</Dialog.Title>
			<Dialog.Description class="text-sm text-muted-foreground">
				Add and configure MCP servers to enable agentic tool execution capabilities.
			</Dialog.Description>
		</div>

		<div class="flex-1 overflow-y-auto p-4 md:p-6">
			<McpSettingsSection {localConfig} onConfigChange={handleConfigChange} />
		</div>

		<div class="flex items-center justify-end gap-3 border-t p-4 md:p-6">
			<Button variant="outline" onclick={handleCancel}>Cancel</Button>
			<Button onclick={handleSave}>Save Changes</Button>
		</div>
	</Dialog.Content>
</Dialog.Root>
