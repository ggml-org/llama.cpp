<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { 
		Cog6ToothIcon,
		SignalIcon,
		ServerIcon,
		ExclamationTriangleIcon
	} from '@heroicons/svelte/24/outline';

	// Mock server status - will be replaced with real store data
	let serverStatus = {
		connected: true,
		model: 'llama-3.2-3b-instruct',
		buildInfo: 'llama.cpp (build 1234)',
		contextSize: 4096,
		loading: false
	};

	let showSettings = false;

	function toggleSettings() {
		showSettings = !showSettings;
		console.log('Toggle settings:', showSettings);
		// TODO: Implement settings dialog
	}

	function getStatusColor() {
		if (serverStatus.loading) return 'bg-yellow-500';
		if (serverStatus.connected) return 'bg-green-500';
		return 'bg-red-500';
	}

	function getStatusText() {
		if (serverStatus.loading) return 'Loading...';
		if (serverStatus.connected) return 'Connected';
		return 'Disconnected';
	}
</script>

<header class="bg-background border-b p-4 flex items-center justify-between">
	<div class="flex items-center space-x-4">
		<h1 class="text-xl font-semibold">llama.cpp</h1>
		
		<!-- Server Status -->
		<div class="flex items-center space-x-2">
			<div class="flex items-center space-x-2">
				<div class="w-2 h-2 rounded-full {getStatusColor()}"></div>
				<span class="text-sm text-muted-foreground">{getStatusText()}</span>
			</div>
			
			{#if serverStatus.connected}
				<Badge variant="outline" class="text-xs">
					<ServerIcon class="h-3 w-3 mr-1" />
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
				<ExclamationTriangleIcon class="h-4 w-4 mr-2" />
				Connection Error
			</Button>
		{/if}
		
		<Button variant="ghost" size="sm" onclick={toggleSettings}>
			<Cog6ToothIcon class="h-4 w-4" />
		</Button>
	</div>
</header>
