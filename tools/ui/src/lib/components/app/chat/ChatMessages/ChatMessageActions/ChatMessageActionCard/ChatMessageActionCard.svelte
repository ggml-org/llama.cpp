<script lang="ts">
	import { ICON_CLASS_DEFAULT } from '$lib/constants/css-classes';
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
		<IconComponent class="{ICON_CLASS_DEFAULT} shrink-0 text-muted-foreground" />
		<span>
			{@render message()}
		</span>
	</div>
	<div class="flex flex-wrap items-center gap-2">
		{@render actions()}
	</div>
</div>
