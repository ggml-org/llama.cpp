<script lang="ts">
	import '../../../src/app.css';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
	import PerfDock from '$lib/components/app/perf/PerfDock.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { onUnderClick = undefined, showUnder = false } = $props();
</script>

<!-- the chat column next to an open sidebar (md:w-72), with the dock as in +layout.svelte -->
<Tooltip.Provider>
	<div class="flex flex-col md:flex-row">
		<div class="hidden w-72 shrink-0 md:block" data-testid="sidebar"></div>

		<div class="min-w-0 flex-1">
			<ChatScreen />
		</div>
	</div>

	{#if showUnder}
		<button
			class="fixed right-0 bottom-0 h-96 w-96"
			data-testid="under"
			onclick={onUnderClick}
			type="button">under</button
		>
	{/if}

	<PerfDock />
</Tooltip.Provider>
