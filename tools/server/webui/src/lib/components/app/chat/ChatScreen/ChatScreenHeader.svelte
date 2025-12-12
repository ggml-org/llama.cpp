<script lang="ts">
	import { Settings } from '@lucide/svelte';
	import { DialogChatSettings } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';

	let settingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}
</script>

<header
	class="md:background-transparent pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center justify-end bg-background/40 p-4 backdrop-blur-xl md:left-[var(--sidebar-width)]"
>
	<div class="pointer-events-auto flex items-center space-x-2">
		<Button variant="ghost" size="sm" onclick={toggleSettings}>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
