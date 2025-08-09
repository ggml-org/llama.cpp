<script lang="ts">
	import { Input } from '$lib/components/ui/input';
	import { Search } from '@lucide/svelte';

	interface Props {
		value?: string;
		placeholder?: string;
		onInput?: (value: string) => void;
	}

	let { value = $bindable(''), placeholder = 'Search conversations...', onInput }: Props = $props();

	function handleInput(event: Event) {
		const target = event.target as HTMLInputElement;
		value = target.value;
		onInput?.(target.value);
	}
</script>

<div class="relative mb-4">
	<Search
		class="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 transform"
	/>
	<Input bind:value {placeholder} class="pl-10" oninput={handleInput} />
</div>
