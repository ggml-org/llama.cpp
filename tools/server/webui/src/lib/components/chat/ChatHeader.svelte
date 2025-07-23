<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Settings } from '@lucide/svelte';
	import ChatSettingsDialog from './ChatSettingsDialog.svelte';
	import ServerStatus from './ServerStatus.svelte';

	let settingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}
</script>

<header class="bg-background flex items-center justify-between border-b p-4">
	<ServerStatus showActions={true} />

	<!-- Actions -->
	<div class="flex items-center space-x-2">
		<Button variant="ghost" size="sm" onclick={toggleSettings}>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<ChatSettingsDialog open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
