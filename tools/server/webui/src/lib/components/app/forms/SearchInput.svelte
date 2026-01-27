<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Search, X } from '@lucide/svelte';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		onClose?: () => void;
		onKeyDown?: (event: KeyboardEvent) => void;
		class?: string;
		id?: string;
		ref?: HTMLInputElement | null;
	}

	let {
		value = $bindable(''),
		placeholder = 'Search...',
		onInput,
		onClose,
		onKeyDown,
		class: className,
		id,
		ref = $bindable(null)
	}: Props = $props();

	let showClearButton = $derived(!!value || !!onClose);

	function handleInput(event: Event) {
		const target = event.target as HTMLInputElement;

		value = target.value;
		onInput?.(target.value);
	}

	function handleClear() {
		if (value) {
			value = '';
			onInput?.('');
			ref?.focus();
		} else {
			onClose?.();
		}
	}
</script>

<div class="relative {className}">
	<Search
		class="absolute top-1/2 left-3 z-10 h-4 w-4 -translate-y-1/2 transform text-muted-foreground"
	/>

	<Input
		{id}
		bind:value
		bind:ref
		class="pl-9 {showClearButton ? 'pr-9' : ''}"
		oninput={handleInput}
		onkeydown={onKeyDown}
		{placeholder}
		type="search"
	/>

	{#if showClearButton}
		<button
			type="button"
			class="absolute top-1/2 right-3 -translate-y-1/2 transform text-muted-foreground transition-colors hover:text-foreground"
			onclick={handleClear}
			aria-label={value ? 'Clear search' : 'Close'}
		>
			<X class="h-4 w-4" />
		</button>
	{/if}
</div>
