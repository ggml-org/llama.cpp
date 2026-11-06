<!--
	The product in a piece: the Llama app in the menu bar, or over the taskbar on Windows, as it is (AppPanel), at a
	size in units. The real UI, never
	a mock-up of it.
-->
<script lang="ts">
	import AppPanel from './AppPanel.svelte';

	interface Props {
		platform?: 'mac' | 'windows';
		/** the panel's width in units */
		size?: number;
		/** the format's unit, in pixels */
		unit: number;
	}

	let { platform = 'mac', size = 14, unit }: Props = $props();
</script>

<!-- the panel is drawn at 380px; zoom takes it to `size` units whatever the format -->
<div style:zoom={(unit * size) / 380} class="shot"><AppPanel {platform} /></div>

<style>
	.shot {
		width: 380px;
	}
</style>
