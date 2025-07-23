<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { AlertTriangle, Server } from '@lucide/svelte';

	interface Props {
		class?: string;
		variant?: 'header' | 'inline';
		showActions?: boolean;
	}

	let { class: className = '', variant = 'header', showActions = false }: Props = $props();

	// Mock server status - will be replaced with real store data
	let serverStatus = {
		connected: true,
		loading: false,
		model: 'llama-3.2-3b-instruct',
		contextSize: 8192,
		tokensPerSecond: 45.2
	};

	function getStatusColor() {
		if (serverStatus.loading) return 'bg-yellow-500';
		if (serverStatus.connected) return 'bg-green-500';
		return 'bg-red-500';
	}

	function getStatusText() {
		return serverStatus.connected ? 'Connected' : 'Disconnected';
	}
</script>

<div class="flex items-center space-x-2 {className}">
	<!-- Status Indicator -->
	<div class="flex items-center space-x-2">
		<div class="h-2 w-2 rounded-full {getStatusColor()}"></div>
		<span class="text-muted-foreground text-sm">{getStatusText()}</span>
	</div>

	<!-- Server Info -->
	{#if serverStatus.connected}
		<Badge variant="outline" class="text-xs">
			<Server class="mr-1 h-3 w-3" />
			{serverStatus.model}
		</Badge>
		<Badge variant="secondary" class="text-xs">
			ctx: {serverStatus.contextSize}
		</Badge>
	{/if}

	<!-- Error Action (if needed) -->
	{#if showActions && !serverStatus.connected}
		<Button variant="outline" size="sm" class="text-destructive">
			<AlertTriangle class="mr-2 h-4 w-4" />
			Connection Error
		</Button>
	{/if}
</div>
