<script lang="ts">
	import { SearchInput } from '$lib/components/app';

	interface Props {
		value?: string;
		/** Search callback; the parent owns debouncing. */
		onSearch?: (query: string) => void;
		placeholder?: string;
	}

	let {
		onSearch,
		placeholder = 'Search models on Hugging Face',
		value = $bindable('')
	}: Props = $props();

	function handleInput(next: string) {
		value = next;

		onSearch?.(value);
	}
</script>

<!-- Inset from the panel edges so the focus ring has room inside the scroll
		container, and opaque so rows scroll behind the bar instead of through it.
		Padding (not margin) for the top room: a sticky margin keeps a gap when
		stuck, where rows would peek through. -->
<div class="sticky top-0 z-99 pt-1.5">
	<SearchInput bind:value onInput={(v) => handleInput(v)} {placeholder} size="sm" />
</div>
