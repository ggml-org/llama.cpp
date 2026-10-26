<script lang="ts">
	interface Props {
		label: string;
		value: string;
	}

	let { label, value }: Props = $props();
</script>

<!-- Metadata chip: label | value pair, matching the HF model page style, on the
     raised surface the tertiary button variant uses -->
<span
	class="inline-flex items-center divide-x divide-border rounded-md border border-border/30 bg-muted/60 text-xs shadow-sm dark:border-border/20 dark:bg-muted/75"
>
	<span class="px-2.5 py-1 text-muted-foreground">{label}</span>

	<span class="px-2.5 py-1 font-medium">{value}</span>
</span>
