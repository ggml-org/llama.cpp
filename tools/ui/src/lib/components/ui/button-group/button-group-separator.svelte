<script lang="ts">
	import { cn } from '$lib/components/ui/utils';
	import type { HTMLAttributes } from 'svelte/elements';

	let { class: className, ...restProps }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div
	class={cn('w-px shrink-0 self-stretch bg-(--button-outline-border)', className)}
	data-slot="button-group-separator"
	{...restProps}
></div>
