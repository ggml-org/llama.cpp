<script lang="ts">
	interface Props {
		/** Left, muted part of the chip, e.g. `Context`. */
		label: string;
		/** Right, emphasized part of the chip, e.g. `262 144`. */
		value: string;
	}

	let { label, value }: Props = $props();
</script>

<!-- Metadata chip: label | value pair, matching the HF model page style -->
<span class="inline-flex items-center divide-x divide-border rounded-md border text-xs">
	<span class="px-2.5 py-1 text-muted-foreground">{label}</span>

	<span class="px-2.5 py-1 font-medium">{value}</span>
</span>
