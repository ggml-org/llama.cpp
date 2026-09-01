<script lang="ts">
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		label: string;
		disabled: boolean;
		onclick: () => void;
	}

	let { label, disabled, onclick }: Props = $props();
</script>

<Button class="w-full transition-transform active:scale-[0.98]" {disabled} {onclick}>
	<Download class="h-4 w-4" />

	{label}
</Button>
