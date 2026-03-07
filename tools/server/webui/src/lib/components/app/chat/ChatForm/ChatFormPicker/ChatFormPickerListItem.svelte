<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		isSelected?: boolean;
		onClick: () => void;
		dataIndex?: number;
		children: Snippet;
	}

	let { isSelected = false, onClick, dataIndex, children }: Props = $props();
</script>

<button
	type="button"
	data-picker-index={dataIndex}
	onclick={onClick}
	class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent/50 {isSelected
		? 'bg-accent/50'
		: ''}"
>
	{@render children()}
</button>
