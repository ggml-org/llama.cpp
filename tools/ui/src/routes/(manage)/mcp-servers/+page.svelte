<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';

	import { Button } from '$lib/components/ui/button';
	import { SIDEBAR_ACTIONS_ITEMS } from '$lib/constants/ui';
	import { SettingsMcpServers, ManageLayout } from '$lib/components/app';
	import { DialogMcpServerAddNew } from '$lib/components/app/dialogs';

	let isAddingServer = $state(false);

	onMount(() => {
		if (page.url.searchParams.has('add')) {
			isAddingServer = true;

			const newUrl = new URL(page.url);
			newUrl.searchParams.delete('add');

			replaceState(newUrl, {});
		}
	});
</script>

<ManageLayout title="MCP Servers">
	{#snippet icon()}
		{@const Icon = SIDEBAR_ACTIONS_ITEMS.find((i) => i.tooltip === 'MCP Servers')?.icon}
		{#if Icon}
			<Icon class="h-5 w-5 md:h-6 md:w-6" />
		{/if}
	{/snippet}

	{#snippet actions()}
		<Button variant="outline" size="lg" onclick={() => (isAddingServer = true)}>
			<Plus class="h-4 w-4" />
			Add New Server
		</Button>
	{/snippet}

	<SettingsMcpServers />

	<DialogMcpServerAddNew bind:open={isAddingServer} />
</ManageLayout>
