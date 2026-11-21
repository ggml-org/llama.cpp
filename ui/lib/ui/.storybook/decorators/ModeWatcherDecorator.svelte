<script lang="ts">
	import { ModeWatcher } from 'mode-watcher';
	import { onMount } from 'svelte';
	import { applyColorScheme } from '../color-scheme';

	interface Props {
		children?: any;
		/** a mode the story is always shown in, rather than the viewer's */
		forced?: 'dark' | 'light';
	}

	let { children, forced }: Props = $props();

	onMount(() => {
		if (forced) document.documentElement.classList.toggle('dark', forced === 'dark');
		else applyColorScheme();
	});
</script>

{#if !forced}<ModeWatcher />{/if}

{#if children}
	{@const Component = children}

	<Component />
{/if}
