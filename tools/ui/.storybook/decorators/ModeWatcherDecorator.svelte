<script lang="ts">
	import { ModeWatcher } from 'mode-watcher';
	import { onMount } from 'svelte';
	import { applyColorScheme } from '../color-scheme';

	interface Props {
		children?: any;
	}

	let { children }: Props = $props();

	onMount(applyColorScheme);
</script>

<ModeWatcher />

{#if children}
	{@const Component = children}

	<Component />
{/if}
