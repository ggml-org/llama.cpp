<script lang="ts">
	import { Server } from '@lucide/svelte';
	import { ServerStatus } from '$lib/components/app';
	import { fade } from 'svelte/transition';

	interface Props {
		/**
		 * Custom loading message
		 */
		message?: string;
		/**
		 * Additional CSS classes
		 */
		class?: string;
	}

	let { 
		message = 'Initializing connection to llama.cpp server...', 
		class: className = '' 
	}: Props = $props();
</script>

<div class="flex h-full items-center justify-center {className}">
	<div class="text-center">
		<div class="mb-4" in:fade={{ duration: 300 }}>
			<div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
				<Server class="h-8 w-8 text-muted-foreground animate-pulse" />
			</div>
			<h2 class="mb-2 text-xl font-semibold">Connecting to Server</h2>
			<p class="text-muted-foreground text-sm">
				{message}
			</p>
		</div>
		<div class="mt-4">
			<ServerStatus class="justify-center" />
		</div>
	</div>
</div>
