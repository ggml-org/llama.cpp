<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Settings } from '@lucide/svelte';
	import { ChatSettingsDialog } from '$lib/components';

	let settingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}
</script>

<header
	class="md:background-transparent bg-background/40 pointer-events-none fixed left-0 right-0 top-0 z-50 flex w-full items-center justify-end p-4 backdrop-blur-xl"
>
	<div class="pointer-events-auto flex items-center space-x-2">
		<Button variant="ghost" size="sm" onclick={toggleSettings}>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<ChatSettingsDialog open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
