<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		isSelected?: boolean;
		disabled?: boolean;
		onclick: () => void;
		onmouseenter?: () => void;
		dataIndex?: number;
		children: Snippet;
		class?: string;
	}

	let {
		children,
		class: className = '',
		dataIndex,
		disabled = false,
		isSelected = false,
		onclick,
		onmouseenter
	}: Props = $props();
</script>

<button
	type="button"
	data-picker-index={dataIndex}
	{disabled}
	{onclick}
	{onmouseenter}
	class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent/50 {isSelected
		? 'bg-accent/50'
		: ''} {disabled ? 'cursor-not-allowed opacity-50' : ''} {className}"
>
	{@render children()}
</button>
