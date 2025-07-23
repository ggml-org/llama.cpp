<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import ChatSidebar from '$lib/components/ChatSidebar.svelte';

	let { children } = $props();
</script>

<ModeWatcher />

<div class="bg-background flex h-screen">
	<!-- Sidebar -->
	<aside class="bg-card w-64 flex-shrink-0 border-r">
		<ChatSidebar />
	</aside>

	<!-- Main Content -->
	<main class="flex flex-1 flex-col overflow-hidden">
		{@render children?.()}
	</main>
</div>
