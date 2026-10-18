<script lang="ts">
	import PerfPanel from './PerfPanel.svelte';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { deviceStore } from '$lib/stores/device.svelte';
	import { perfStore } from '$lib/stores/perf.svelte';
	import { settingsStore } from '$lib/stores/settings/index.svelte';
	import { untrack } from 'svelte';

	// the store owns the polling of /perf, the setting only switches it
	$effect(() => {
		const on = Boolean(settingsStore.config[SETTINGS_KEYS.DISPLAY_PERFORMANCE]);

		untrack(() => perfStore.setEnabled(on));
	});
</script>

<!-- wide screens: a card in the right margin, clear of the composer; narrower ones get the strip in ChatScreen -->
{#if perfStore.enabled && deviceStore.isWide}
	<div class="pointer-events-none fixed right-4 bottom-10 z-40 w-52">
		<PerfPanel />
	</div>
{/if}
