<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatFormActionRecord from '#lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionRecord.svelte';

	let {
		hasAudioModality = true,
		micWarning = null,
		onMicClick
	}: { hasAudioModality?: boolean; micWarning?: string | null; onMicClick?: () => void } = $props();
</script>

<Tooltip.Provider>
	<ChatFormActionRecord {hasAudioModality} {micWarning} {onMicClick} />
</Tooltip.Provider>
