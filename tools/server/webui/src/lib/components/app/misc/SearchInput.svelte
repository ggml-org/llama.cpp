<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Search } from '@lucide/svelte';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
		class?: string;
		id?: string;
		ref?: HTMLInputElement | null;
	}

	let {
		value = $bindable(''),
		placeholder = 'Search...',
		onInput,
		class: className,
		id,
		ref = $bindable(null)
	}: Props = $props();

	function handleInput(event: Event) {
		const target = event.target as HTMLInputElement;

		value = target.value;
		onInput?.(target.value);
	}
</script>

<div class="relative {className}">
	<Search
		class="absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 transform text-muted-foreground"
	/>

	<Input {id} bind:value bind:ref class="pl-9" oninput={handleInput} {placeholder} type="search" />
</div>
