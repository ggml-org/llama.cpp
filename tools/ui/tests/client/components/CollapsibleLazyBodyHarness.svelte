<script lang="ts">
	import CollapsibleContentBlock from '$lib/components/app/content/CollapsibleContentBlock.svelte';
	import CollapsibleTerminalBlock from '$lib/components/app/content/CollapsibleTerminalBlock.svelte';

	interface Props {
		variant: 'content' | 'terminal';
		open: boolean;
	}

	let { variant, open }: Props = $props();
</script>

{#if variant === 'content'}
	<CollapsibleContentBlock {open} title="Block">
		<span>collapsible-body-marker</span>
	</CollapsibleContentBlock>
{:else}
	<CollapsibleTerminalBlock {open} title="Block">
		<span>collapsible-body-marker</span>
	</CollapsibleTerminalBlock>
{/if}
