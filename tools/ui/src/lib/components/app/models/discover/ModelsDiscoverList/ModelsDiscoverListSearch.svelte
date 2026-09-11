<script lang="ts">
	import { SearchInput } from '$lib/components/app';

	interface Props {
		value?: string;
		/** Search callback; the parent owns debouncing. */
		onSearch?: (query: string) => void;
		placeholder?: string;
	}

	let { onSearch, placeholder = 'Search models...', value = $bindable('') }: Props = $props();

	function handleInput(next: string) {
		value = next;

		onSearch?.(value);
	}
</script>

<div class="sticky top-0 z-99 p-2">
	<SearchInput bind:value onInput={(v) => handleInput(v)} {placeholder} />
</div>
