<script lang="ts">
	import { ModeWatcher } from 'mode-watcher';
	import { onMount } from 'svelte';
	import { ColorMode } from '#lib/enums/ui.enums.ts';
	import { applyColorScheme, setDarkClass } from '../color-scheme';

	interface Props {
		children?: any;
		/** a mode the story is always shown in, rather than the viewer's */
		forced?: ColorMode.DARK | ColorMode.LIGHT;
	}

	let { children, forced }: Props = $props();

	onMount(() => {
		if (forced) setDarkClass(forced === ColorMode.DARK);
		else applyColorScheme();
	});
</script>

{#if !forced}<ModeWatcher />{/if}

{#if children}
	{@const Component = children}

	<Component />
{/if}
