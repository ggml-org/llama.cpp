<script lang="ts">
	interface Props {
		label: string;
		value: string;
		subtitle?: string;
	}

	let { label, value, subtitle }: Props = $props();
</script>

<div class="grid gap-1.5">
	<div class="flex items-baseline justify-between">
		<span class="text-muted-foreground">{label}</span>
		<span class="font-mono text-muted-foreground">{value}</span>
	</div>

	{#if subtitle}
		<div class="text-[10px] leading-tight text-muted-foreground/70">{subtitle}</div>
	{/if}
</div>
