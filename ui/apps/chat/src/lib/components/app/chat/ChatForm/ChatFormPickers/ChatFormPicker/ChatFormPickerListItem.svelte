<script lang="ts">
	import { UI_DATA_ATTRS } from '#lib/constants/index.ts';
	import { hasPointerMoved } from '#lib/utils/index.ts';
	import type { Snippet } from 'svelte';

	interface Props {
		isSelected?: boolean;
		disabled?: boolean;
		onclick: () => void;
		/** Called when the user moves the pointer over the row, not when the row renders under it. */
		onhover?: () => void;
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
		onhover
	}: Props = $props();
</script>

<button
	{disabled}
	{onclick}
	onmousemove={(event) => {
		if (hasPointerMoved(event)) onhover?.();
	}}
	type="button"
	{...{ [UI_DATA_ATTRS.PICKER_INDEX]: dataIndex }}
	class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent/50 {isSelected
		? 'bg-accent/50'
		: ''} {disabled ? 'cursor-not-allowed opacity-50' : ''} {className}"
>
	{@render children()}
</button>
