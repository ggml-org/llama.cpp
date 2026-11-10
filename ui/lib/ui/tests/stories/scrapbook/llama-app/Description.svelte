<!-- A catalog description, its **phrase** in the default ink and the rest muted, as llama.app sets it -->
<script lang="ts">
	let { class: className = '', text }: { class?: string; text: string } = $props();
</script>

<p class={className}>
	{#each text.split(/(\*\*[^*]+\*\*)/g) as part, i (i)}
		{#if part.startsWith('**') && part.endsWith('**')}
			<strong class="font-medium text-foreground">{part.slice(2, -2)}</strong>
		{:else}
			{part}
		{/if}
	{/each}
</p>
