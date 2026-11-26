<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatMessageStatistics from '#lib/components/app/chat/ChatMessages/ChatMessageStatistics/ChatMessageStatistics.svelte';
	import type { ComponentProps } from 'svelte';

	let props: ComponentProps<typeof ChatMessageStatistics> = $props();
</script>

<Tooltip.Provider>
	<ChatMessageStatistics {...props} />
</Tooltip.Provider>
