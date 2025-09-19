<script lang="ts">
	import { ModeWatcher } from 'mode-watcher';
	import { onMount } from 'svelte';

	interface Props {
		children?: any;
	}

	let { children }: Props = $props();

	onMount(() => {
		const root = document.documentElement;
		const theme = localStorage.getItem('mode-watcher-mode') || 'system';

		if (theme === 'dark') {
			root.classList.add('dark');
		} else if (theme === 'light') {
			root.classList.remove('dark');
		} else {
			const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
			if (prefersDark) {
				root.classList.add('dark');
			} else {
				root.classList.remove('dark');
			}
		}
	});
</script>

<ModeWatcher />

{#if children}
	{@const Component = children}

	<Component />
{/if}
