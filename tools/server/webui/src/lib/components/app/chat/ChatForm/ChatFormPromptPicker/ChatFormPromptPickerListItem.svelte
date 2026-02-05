<script lang="ts">
	import type { MCPPromptInfo, MCPServerSettingsEntry } from '$lib/types';
	import ChatFormPromptPickerHeader from './ChatFormPromptPickerHeader.svelte';

	interface Props {
		prompt: MCPPromptInfo;
		server: MCPServerSettingsEntry | undefined;
		serverLabel: string;
		isSelected?: boolean;
		onClick: () => void;
		'data-prompt-index'?: number;
	}

	let {
		prompt,
		server,
		serverLabel,
		isSelected = false,
		onClick,
		'data-prompt-index': dataPromptIndex
	}: Props = $props();
</script>

<button
	type="button"
	data-prompt-index={dataPromptIndex}
	onclick={onClick}
	class="flex w-full cursor-pointer items-start gap-3 rounded-lg px-3 py-2 text-left hover:bg-accent/50 {isSelected
		? 'bg-accent/50'
		: ''}"
>
	<ChatFormPromptPickerHeader {prompt} {server} {serverLabel} />
</button>
