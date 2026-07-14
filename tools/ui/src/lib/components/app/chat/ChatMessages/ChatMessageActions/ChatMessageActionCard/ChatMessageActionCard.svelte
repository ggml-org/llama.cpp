<script lang="ts">
	import type { Snippet, Component } from 'svelte';

	interface Props {
		icon: Component<{ class?: string }>;
		message: Snippet;
		actions: Snippet;
	}

	let { icon: IconComponent, message, actions }: Props = $props();
</script>

<div class="my-2 rounded-lg border border-border bg-card p-3">
	<div class="mb-3 flex items-center gap-2 text-sm">
		<IconComponent class="h-4 w-4 shrink-0 text-muted-foreground" />
		<span>
			{@render message()}
		</span>
	</div>
	<div class="flex flex-wrap items-center gap-2">
		{@render actions()}
	</div>
</div>
