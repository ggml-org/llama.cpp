<script lang="ts">
	import { SearchInput } from '$lib/components/app';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		class?: string;
	}

	let {
		value = $bindable(''),
		placeholder = 'Search conversations...',
		onInput,
		class: className
	}: Props = $props();
</script>

<SearchInput bind:value {placeholder} {onInput} class="mb-4 {className}" />
