<script lang="ts">
	interface Props {
		expanded: boolean;
		onToggle: () => void;
	}

	let { expanded, onToggle }: Props = $props();
</script>

<!-- sits after the section's last row, sized like one: "Show more" opens the
     whole section, "Show less" closes it back to its first rows -->
<button
	class="flex min-h-9 w-full cursor-pointer items-center rounded-lg px-2 py-1.5 text-left text-sm font-medium text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground"
	onclick={onToggle}
	type="button"
>
	{expanded ? 'Show less' : 'Show more'}
</button>
