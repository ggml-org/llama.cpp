<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import ChatSidebar from '$lib/components/ChatSidebar.svelte';

	let { children } = $props();
</script>

<ModeWatcher />

<div class="flex h-screen bg-background">
	<!-- Sidebar -->
	<aside class="w-80 border-r bg-card flex-shrink-0">
		<ChatSidebar />
	</aside>

	<!-- Main Content -->
	<main class="flex-1 flex flex-col overflow-hidden">
		{@render children?.()}
	</main>
</div>
