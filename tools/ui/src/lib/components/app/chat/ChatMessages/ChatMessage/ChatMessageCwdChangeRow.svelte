<script lang="ts">
	import { Folder } from '@lucide/svelte';

	interface Props {
		class?: string;
		/** Abbreviated path, e.g. `~/git/project`. */
		display: string;
		/** Absolute path, shown on hover. */
		path: string;
	}

	let { class: className = '', display, path }: Props = $props();
</script>

<div class="text-muted-foreground flex items-center gap-2 py-1.5 {className}">
	<Folder class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />

	<span class="text-foreground/80 text-sm font-medium">Set working directory to&nbsp;</span>

	<span class="font-mono text-foreground/90 text-sm break-all" title={path}>{display}</span>
</div>
