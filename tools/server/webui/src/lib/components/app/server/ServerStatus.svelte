<script lang="ts">
	import { AlertTriangle, Server } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { serverProps, serverLoading, serverError, modelName } from '$lib/stores/server.svelte';

	interface Props {
		class?: string;
		showActions?: boolean;
	}

	let { class: className = '', showActions = false }: Props = $props();

	let error = $derived(serverError());
	let loading = $derived(serverLoading());
	let model = $derived(modelName());
	let serverData = $derived(serverProps());

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

<div class="flex items-center space-x-3 {className}">
	<div class="flex items-center space-x-2">
		<div class="h-2 w-2 rounded-full {getStatusColor()}"></div>

		<span class="text-sm text-muted-foreground">{getStatusText()}</span>
	</div>

	{#if serverData && !error}
		<Badge variant="outline" class="text-xs">
			<Server class="mr-1 h-3 w-3" />

			{model || 'Unknown Model'}
		</Badge>

		{#if serverData.default_generation_settings.n_ctx}
			<Badge variant="secondary" class="text-xs">
				ctx: {serverData.default_generation_settings.n_ctx.toLocaleString()}
			</Badge>
		{/if}
	{/if}

	{#if showActions && error}
		<Button variant="outline" size="sm" class="text-destructive">
			<AlertTriangle class="h-4 w-4" />

			{error}
		</Button>
	{/if}
</div>
