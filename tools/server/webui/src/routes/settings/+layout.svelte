<script lang="ts">
	import { X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app';

	let { children } = $props();

	let previousRouteId = $state<string | null>(null);

	$effect(() => {
		const currentId = page.route.id;
		return () => {
			previousRouteId = currentId;
		};
	});

	function handleClose() {
		const prevIsSettings = previousRouteId?.startsWith('/settings');
		if (browser && window.history.length > 1 && !prevIsSettings) {
			history.back();
		} else {
			goto('#/');
		}
	}
</script>

<div class="relative h-full">
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div class="min-h-full">
		{@render children?.()}
	</div>
</div>
