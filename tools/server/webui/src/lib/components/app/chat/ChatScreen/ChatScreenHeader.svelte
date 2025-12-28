// [AI] MCP Settings button integration
<script lang="ts">
	import { Settings, Puzzle } from '@lucide/svelte';
	import { DialogChatSettings, DialogMCPSettings } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';

	const sidebar = useSidebar();

	let settingsOpen = $state(false);
	let mcpSettingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}

	function toggleMCPSettings() {
		mcpSettingsOpen = true;
	}
</script>

<header
	class="md:background-transparent pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center justify-end bg-background/40 p-4 backdrop-blur-xl duration-200 ease-linear {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto flex items-center space-x-2">
		<Button variant="ghost" size="sm" onclick={toggleMCPSettings}>
			<Puzzle class="h-4 w-4" />
		</Button>
		<Button variant="ghost" size="sm" onclick={toggleSettings}>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
<DialogMCPSettings open={mcpSettingsOpen} onOpenChange={(open) => (mcpSettingsOpen = open)} />
