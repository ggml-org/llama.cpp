<script lang="ts">
	import DialogMcpServers from '$lib/components/app/dialogs/DialogMcpServers.svelte';
	import DialogSettingsChat from '$lib/components/app/dialogs/DialogSettingsChat.svelte';

	let { panel }: { panel: 'mcp' | 'settings' } = $props();
</script>

{#if panel === 'settings'}
	<DialogSettingsChat open />
{:else}
	<DialogMcpServers open />
{/if}
