<script lang="ts">
	import { Download } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		label: string;
		disabled: boolean;
		onclick: () => void;
	}

	let { disabled, label, onclick }: Props = $props();
</script>

<Button
	class="h-10 w-full rounded-lg transition-transform active:scale-[0.97]"
	{disabled}
	{onclick}
>
	<Download class="h-4 w-4" />

	{label}
</Button>
