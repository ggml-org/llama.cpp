<script lang="ts">
	import { SearchInput } from '$lib/components/app';

	interface Props {
		value?: string;
		/** Debounced search callback (300ms). */
		onSearch?: (query: string) => void;
		placeholder?: string;
	}

	let { onSearch, placeholder = 'Search models...', value = $bindable('') }: Props = $props();

	let searchTimeout: ReturnType<typeof setTimeout> | null = null;

	function handleInput(next: string) {
		value = next;

		if (searchTimeout) clearTimeout(searchTimeout);

		searchTimeout = setTimeout(() => onSearch?.(value), 300);
	}
</script>

<div class="sticky top-0 z-99 p-2">
	<SearchInput bind:value onInput={(v) => handleInput(v)} {placeholder} />
</div>
