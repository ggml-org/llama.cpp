<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { AlertTriangle, Server, Settings } from '@lucide/svelte';
	import ChatSettingsDialog from './ChatSettingsDialog.svelte';

	// Mock server status - will be replaced with real store data
	let serverStatus = {
		connected: true,
		loading: false,
		model: 'llama-3.2-3b-instruct',
		contextSize: 8192,
		tokensPerSecond: 45.2
	};

	let settingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}

	function getStatusColor() {
		if (serverStatus.loading) return 'bg-yellow-500';
		if (serverStatus.connected) return 'bg-green-500';
		return 'bg-red-500';
	}

	function getStatusText() {
		return serverStatus.connected ? 'Connected' : 'Disconnected';
	}
</script>

<header class="bg-background flex items-center justify-between border-b p-4">
	<div class="flex items-center space-x-4">
		<h1 class="text-xl font-semibold">llama.cpp</h1>

		<!-- Server Status -->
		<div class="flex items-center space-x-2">
			<div class="flex items-center space-x-2">
				<div class="h-2 w-2 rounded-full {getStatusColor()}"></div>
				<span class="text-muted-foreground text-sm">{getStatusText()}</span>
			</div>

			{#if serverStatus.connected}
				<Badge variant="outline" class="text-xs">
					<Server class="mr-1 h-3 w-3" />
					{serverStatus.model}
				</Badge>
				<Badge variant="secondary" class="text-xs">
					ctx: {serverStatus.contextSize}
				</Badge>
			{/if}
		</div>
	</div>

	<!-- Actions -->
	<div class="flex items-center space-x-2">
		{#if !serverStatus.connected}
			<Button variant="outline" size="sm" class="text-destructive">
				<AlertTriangle class="mr-2 h-4 w-4" />
				Connection Error
			</Button>
		{/if}

		<Button variant="ghost" size="sm" onclick={toggleSettings}>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<ChatSettingsDialog open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
