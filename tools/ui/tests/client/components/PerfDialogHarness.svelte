<script lang="ts">
	import '../../../src/app.css';
	import DialogSettingsChat from '$lib/components/app/dialogs/DialogSettingsChat.svelte';
	import PerfDock from '$lib/components/app/perf/PerfDock.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { open = false } = $props();
</script>

<!-- the settings dialog over the dock, as in +layout.svelte (the dialog lives in the sidebar) -->
<Tooltip.Provider>
	<DialogSettingsChat {open} />

	<PerfDock />
</Tooltip.Provider>
