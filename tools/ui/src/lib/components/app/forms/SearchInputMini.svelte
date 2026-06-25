<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Search, X } from '@lucide/svelte';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		onKeyDown?: (event: KeyboardEvent) => void;
		class?: string;
		id?: string;
		autofocus?: boolean;
	}

	let {
		value = $bindable(''),
		placeholder = 'Search...',
		onInput,
		onKeyDown,
		class: className,
		id,
		autofocus = false
	}: Props = $props();

	function handleInput(event: Event) {
		const target = event.target as HTMLInputElement;
		value = target.value;
		onInput?.(target.value);
	}

	function handleKeyDown(event: KeyboardEvent) {
		// Stop keydown from bubbling to the surrounding dropdown menu so the
		// menu's typeahead (single characters), Home/End, and arrow-key
		// navigation don't fire while the user is typing here. Escape and
		// Tab are intentionally left to bubble so the dropdown can still
		// close on Escape and move focus on Tab.
		if (event.key !== 'Escape' && event.key !== 'Tab') {
			event.stopPropagation();
		}
		onKeyDown?.(event);
	}

	function handleClear(event: MouseEvent) {
		event.stopPropagation();
		value = '';
		onInput?.('');
	}
</script>

<div class="relative {className}">
	<Search
		class="pointer-events-none absolute top-1/2 left-2 z-10 h-3 w-3 -translate-y-1/2 transform text-muted-foreground"
	/>

	<Input
		{autofocus}
		{id}
		bind:value
		class="h-7 pl-7 text-xs {value ? 'pr-7' : ''}"
		oninput={handleInput}
		onkeydown={handleKeyDown}
		{placeholder}
		type="search"
	/>

	{#if value}
		<button
			type="button"
			class="absolute top-1/2 right-1.5 z-10 flex h-5 w-5 -translate-y-1/2 transform cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
			onclick={handleClear}
			aria-label="Clear search"
		>
			<X class="h-3 w-3" />
		</button>
	{/if}
</div>
