<script lang="ts">
	import { X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		id: string;
		onRemove?: (id: string) => void;
		class?: string;
		iconSize?: number;
	}

	let { id, onRemove, class: className = '', iconSize = 3 }: Props = $props();
</script>

<Button
	type="button"
	variant="ghost"
	size="icon-sm"
	class="bg-white/20 p-0 hover:bg-white/30 {className}"
	onclick={(e: MouseEvent) => {
		e.stopPropagation();
		onRemove?.(id);
	}}
	aria-label="Remove file"
>
	<X class="h-{iconSize} w-{iconSize}" />
</Button>
