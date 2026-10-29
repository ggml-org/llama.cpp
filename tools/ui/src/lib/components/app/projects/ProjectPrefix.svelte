<script lang="ts">
	import { PROJECT_ICON } from '$lib/constants';

	interface Props {
		class?: string;
		/** Name of the project the item belongs to. */
		name: string;
	}

	let { class: className = '', name }: Props = $props();
</script>

<!-- Leads a chat title with the project it belongs to: icon, name, slash.
     Callers cap its width; the title next to it truncates first. -->
<span class="inline-flex min-w-0 shrink items-center gap-1 text-muted-foreground {className}">
	<PROJECT_ICON aria-hidden="true" class="h-3.5 w-3.5 shrink-0" />

	<span class="truncate">{name}</span>

	<span aria-hidden="true" class="shrink-0">/</span>
</span>
