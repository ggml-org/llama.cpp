<script lang="ts">
	import { AlertTriangle, Server } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { modelsStore, serverStore } from '$lib/stores';

	interface Props {
		class?: string;
		showActions?: boolean;
	}

	let { class: className = '', showActions = false }: Props = $props();

	let error = $derived(serverStore.error);
	let loading = $derived(serverStore.loading);
	let model = $derived(modelsStore.singleModelName);
	let serverData = $derived(serverStore.props);

	function getStatusColor() {
		if (loading) return 'status-lamp orb-warning';

		if (error) return 'status-lamp orb-danger';

		if (serverData) return 'status-lamp orb-success';

		return 'bg-gray-500';
	}

	function getStatusText() {
		if (loading) return 'Connecting...';

		if (error) return 'Connection Error';

		if (serverData) return 'Connected';

		return 'Unknown';
	}
</script>

<div class="flex items-center space-x-3 {className}">
	<div class="flex items-center space-x-2">
		<div class="size-2 rounded-full {getStatusColor()}"></div>

		<span class="text-sm text-muted-foreground">{getStatusText()}</span>
	</div>

	{#if serverData && !error}
		<Badge class="text-xs" variant="outline">
			<Server class="mr-1 size-icon-xs" />

			{model || 'Unknown Model'}
		</Badge>

		{#if serverData?.default_generation_settings?.n_ctx}
			<Badge class="text-xs" variant="secondary">
				ctx: {serverData.default_generation_settings.n_ctx.toLocaleString()}
			</Badge>
		{/if}
	{/if}

	{#if showActions && error}
		<Button class="text-destructive" size="sm" variant="outline">
			<AlertTriangle class={ICON_CLASS_DEFAULT} />

			{error}
		</Button>
	{/if}
</div>
