<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { AlertTriangle, Server } from '@lucide/svelte';
	import { serverProps, serverLoading, serverError, modelName } from '$lib/stores/server.svelte';

	interface Props {
		class?: string;
		variant?: 'header' | 'inline';
		showActions?: boolean;
	}

	let { class: className = '', variant = 'header', showActions = false }: Props = $props();

	// Real server status from store
	const serverData = $derived(serverProps());
	const loading = $derived(serverLoading());
	const error = $derived(serverError());
	const model = $derived(modelName());

	function getStatusColor() {
		if (loading) return 'bg-yellow-500';
		if (error) return 'bg-red-500';
		if (serverData) return 'bg-green-500';
		return 'bg-gray-500';
	}

	function getStatusText() {
		if (loading) return 'Connecting...';
		if (error) return 'Connection Error';
		if (serverData) return 'Connected';
		return 'Unknown';
	}
</script>

<div class="flex items-center space-x-2 {className}">
	<!-- Status Indicator -->
	<div class="flex items-center space-x-2">
		<div class="h-2 w-2 rounded-full {getStatusColor()}"></div>
		<span class="text-muted-foreground text-sm">{getStatusText()}</span>
	</div>

	<!-- Server Info -->
	{#if serverData && !error}
		<Badge variant="outline" class="text-xs">
			<Server class="mr-1 h-3 w-3" />
			{model || 'Unknown Model'}
		</Badge>
		{#if serverData.n_ctx}
			<Badge variant="secondary" class="text-xs">
				ctx: {serverData.n_ctx.toLocaleString()}
			</Badge>
		{/if}
	{/if}

	<!-- Error Action (if needed) -->
	{#if showActions && error}
		<Button variant="outline" size="sm" class="text-destructive">
			<AlertTriangle class="mr-2 h-4 w-4" />
			{error}
		</Button>
	{/if}
</div>
