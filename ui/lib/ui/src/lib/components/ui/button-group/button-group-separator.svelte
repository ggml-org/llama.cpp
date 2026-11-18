<script lang="ts">
	import { cn } from '../utils';
	import type { HTMLAttributes } from 'svelte/elements';

	let { class: className, ...restProps }: HTMLAttributes<HTMLDivElement> = $props();
</script>

<div
	class={cn('w-px shrink-0 self-stretch bg-(--button-group-separator)', className)}
	data-slot="button-group-separator"
	{...restProps}
></div>
