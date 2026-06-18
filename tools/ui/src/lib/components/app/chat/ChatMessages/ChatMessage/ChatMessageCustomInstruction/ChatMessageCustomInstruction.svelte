<script lang="ts">
	import { BookText } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';

	interface Props {
		class?: string;
		instructionId: string;
		title: string;
	}

	let { class: className = '', instructionId, title }: Props = $props();
</script>

<div
	aria-label={`Custom instruction: ${title}`}
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	<div class="max-w-[80%]">
		<Card
			class="overflow-y-auto rounded-[1.125rem] !border-2 !border-dashed !border-border/50 bg-muted px-3.75 py-1.5"
		>
			<div class="flex items-center gap-2">
				<BookText class="h-4 w-4 text-muted-foreground" />
				<span class="text-sm font-medium">{title}</span>
			</div>
		</Card>
	</div>

	<div class="flex gap-2">
		<span class="text-xs text-muted-foreground">Custom Instruction</span>
		<span class="text-xs text-muted-foreground">{instructionId.slice(0, 8)}</span>
	</div>
</div>