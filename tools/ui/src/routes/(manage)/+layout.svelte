<script lang="ts">
	import { X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app';
	import { ROUTES } from '$lib/constants';

	let { children } = $props();

	let previousRouteId = $state<string | null>(null);

	$effect(() => {
		const currentId = page.route.id;
		return () => {
			previousRouteId = currentId;
		};
	});

	function handleClose() {
		// If the previous route is the same manage page we're on now
		// (e.g. re-entering via sidebar), history.back() would loop.
		const samePage = previousRouteId === page.route.id;
		if (browser && window.history.length > 1 && !samePage) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}
</script>

<div class="fixed top-4.5 right-4 z-50 md:hidden">
	<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
</div>

{@render children?.()}
