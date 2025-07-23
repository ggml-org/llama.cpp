<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Bot } from '@lucide/svelte';
</script>

<div class="flex justify-start gap-3">
	<div
		class="bg-background flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow"
	>
		<Bot class="h-4 w-4" />
	</div>

	<Card class="bg-muted max-w-[80%] p-3">
		<div class="flex items-center space-x-2">
			<div class="flex space-x-1">
				<div
					class="h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.3s]"
				></div>
				<div
					class="h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.15s]"
				></div>
				<div class="h-2 w-2 animate-bounce rounded-full bg-current"></div>
			</div>
			<span class="text-muted-foreground text-sm">Thinking...</span>
		</div>
	</Card>
</div>
