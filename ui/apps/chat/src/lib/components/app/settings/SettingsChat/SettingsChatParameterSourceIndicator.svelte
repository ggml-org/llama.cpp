<script lang="ts">
	import { Badge } from '@ggml-org/llama-ui/ui/badge';
	import { Wrench } from '@lucide/svelte';

	interface Props {
		class?: string;
	}

	let { class: className = '' }: Props = $props();
</script>

<Badge
	class="h-5 bg-orange-100 px-1.5 py-0.5 text-caption text-orange-800 dark:bg-orange-900 dark:text-orange-200 {className}"
	variant="secondary"
>
	<Wrench class="mr-1 size-icon-xs" />

	Custom
</Badge>
