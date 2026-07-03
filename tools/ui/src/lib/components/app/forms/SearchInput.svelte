<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Search, X } from '@lucide/svelte';

	type Variant = 'default' | 'sm';

	interface Props {
		autofocus?: boolean;
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		onClose?: () => void;
		onKeyDown?: (event: KeyboardEvent) => void;
		class?: string;
		id?: string;
		ref?: HTMLInputElement | null;
		isCancelAlwaysVisible?: boolean;
		variant?: Variant;
	}

	let {
		autofocus,
		value = $bindable(''),
		placeholder = 'Search...',
		onInput,
		onClose,
		onKeyDown,
		class: className,
		id,
		ref = $bindable(null),
		isCancelAlwaysVisible = false,
		variant = 'default'
	}: Props = $props();

	let isSm = $derived(variant === 'sm');

	let showClearButton = $derived(isCancelAlwaysVisible || (!isSm && !!value) || !!onClose);

	const inputClass = $derived.by(() => {
		const sizeClass = isSm ? 'h-7 pl-7 text-xs' : 'pl-9';
		const clearClass = showClearButton ? (isSm ? ' pr-7' : ' pr-9') : '';
		return sizeClass + clearClass;
	});

	function handleInput(event: Event) {
		const target = event.target as HTMLInputElement;

		value = target.value;
		onInput?.(target.value);
	}

	function handleClear() {
		if (value) {
			value = '';
			onInput?.('');
			ref?.focus({ preventScroll: true });
		} else {
			onClose?.();
		}
	}

	function handleKeyDown(event: KeyboardEvent) {
		if (isSm) {
			// Stop keydown from bubbling to the surrounding dropdown menu so the
			// menu's typeahead (single characters), Home/End, and arrow-key
			// navigation don't fire while the user is typing here. Escape and
			// Tab are intentionally left to bubble so the dropdown can still
			// close on Escape and move focus on Tab.
			if (event.key !== 'Escape' && event.key !== 'Tab') {
				event.stopPropagation();
			}
		}

		onKeyDown?.(event);
	}
</script>

<div class="relative {className}">
	<Search
		class={isSm
			? 'pointer-events-none absolute top-1/2 left-2 z-10 h-3 w-3 -translate-y-1/2 transform text-muted-foreground'
			: 'absolute top-1/2 left-3 z-10 h-4 w-4 -translate-y-1/2 transform text-muted-foreground'}
	/>

	<Input
		{autofocus}
		{id}
		bind:value
		bind:ref
		class={inputClass}
		oninput={handleInput}
		onkeydown={handleKeyDown}
		{placeholder}
		type="search"
	/>

	{#if showClearButton}
		<button
			type="button"
			class={isSm
				? 'absolute top-1/2 right-1.5 z-10 flex h-5 w-5 -translate-y-1/2 transform cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground'
				: 'absolute top-1/2 right-3 -translate-y-1/2 transform cursor-pointer text-muted-foreground transition-colors hover:text-foreground'}
			onclick={handleClear}
			aria-label={value ? 'Clear search' : 'Close'}
		>
			<X class={isSm ? 'h-3 w-3' : 'h-4 w-4'} />
		</button>
	{/if}
</div>
