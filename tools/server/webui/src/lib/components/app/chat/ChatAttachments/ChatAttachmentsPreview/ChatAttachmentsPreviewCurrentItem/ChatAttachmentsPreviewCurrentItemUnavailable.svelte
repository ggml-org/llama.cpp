<script lang="ts">
	import type { Component } from 'svelte';

	interface Props {
		IconComponent?: Component;
	}

	let { IconComponent }: Props = $props();
</script>

<div class="flex flex-1 items-center justify-center p-8">
	<div class="text-center">
		{#if IconComponent}
			<IconComponent class="mx-auto mb-4 h-16 w-16 text-white/50" />
		{/if}

		<p class="text-white/70">Preview not available for this file type</p>
	</div>
</div>
