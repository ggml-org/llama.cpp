<script lang="ts">
	import { SearchInput } from '#lib/components/app/index.ts';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		class?: string;
	}

	let {
		class: className,
		onInput,
		placeholder = 'Search conversations...',
		value = $bindable('')
	}: Props = $props();
</script>

<div class="mb-4 px-2 {className}">
	<SearchInput bind:value {onInput} {placeholder} />
</div>
