<script lang="ts">
	import { X } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { ActionIcon } from '$lib/components/app';

	let { children } = $props();

	function handleClose() {
		if (browser && window.history.length > 1) {
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
