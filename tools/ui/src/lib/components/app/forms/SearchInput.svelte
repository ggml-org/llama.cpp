<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
	import { Input } from '$lib/components/ui/input';
	import { Search, X } from '@lucide/svelte';

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
		isCancelAlwaysVisible = false
	}: Props = $props();

	let showClearButton = $derived(isCancelAlwaysVisible || !!value || !!onClose);

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
</script>

<div class="relative {className}">
	<Search
		class="absolute top-1/2 left-3 z-10 {ICON_CLASS_DEFAULT} -translate-y-1/2 transform text-muted-foreground"
	/>

	<Input
		{autofocus}
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
			class="absolute top-1/2 right-3 -translate-y-1/2 transform cursor-pointer text-muted-foreground transition-colors hover:text-foreground"
			onclick={handleClear}
			aria-label={value ? 'Clear search' : 'Close'}
		>
			<X class={ICON_CLASS_DEFAULT} />
		</button>
	{/if}
</div>
