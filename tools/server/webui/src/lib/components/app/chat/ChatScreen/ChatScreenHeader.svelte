<script lang="ts">
	import { Settings } from '@lucide/svelte';
	import { DialogChatSettings } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { useSidebar } from '$lib/components/ui/sidebar';

	const sidebar = useSidebar();

	let settingsOpen = $state(false);

	function toggleSettings() {
		settingsOpen = true;
	}
</script>

<header
	class="pointer-events-none fixed top-0 right-0 left-0 z-50 flex items-center justify-end p-4 duration-200 ease-linear {sidebar.open
		? 'md:left-[var(--sidebar-width)]'
		: ''}"
>
	<div class="pointer-events-auto flex items-center space-x-2">
		<Button
			variant="ghost"
			size="icon"
			onclick={toggleSettings}
			class="rounded-full backdrop-blur-lg"
		>
			<Settings class="h-4 w-4" />
		</Button>
	</div>
</header>

<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
