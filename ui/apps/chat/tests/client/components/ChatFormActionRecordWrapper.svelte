<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatFormActionRecord from '#lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionRecord.svelte';

	let {
		micWarning = null,
		onMicClick,
		unavailableReason = null
	}: {
		micWarning?: string | null;
		onMicClick?: () => void;
		unavailableReason?: string | null;
	} = $props();
</script>

<Tooltip.Provider>
	<ChatFormActionRecord {micWarning} {onMicClick} {unavailableReason} />
</Tooltip.Provider>
