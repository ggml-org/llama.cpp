<script lang="ts">
	import ChatFormActionRecord from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionRecord.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let {
		hasAudioModality = true,
		micWarning = null,
		onMicClick
	}: { hasAudioModality?: boolean; micWarning?: string | null; onMicClick?: () => void } = $props();
</script>

<Tooltip.Provider>
	<ChatFormActionRecord {hasAudioModality} {micWarning} {onMicClick} />
</Tooltip.Provider>
