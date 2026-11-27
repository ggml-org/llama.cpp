<!--
	The phone's Add to chat sheet on its own: the chat form's actions as the sheet reads them, and a
	+ key that opens it.
-->
<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Plus } from '@lucide/svelte';
	import ChatFormActionAddDrawer from '#lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionAddDrawer.svelte';
	import { ATTACHMENT_MENU_TEXT } from '#lib/constants/index.ts';
	import { setChatFormActionsContext } from '#lib/contexts/index.ts';

	interface Props {
		/** The model takes images, so the sheet offers the camera. */
		hasVisionModality?: boolean;
		/** The chat's working directory, abbreviated; null offers to set one. */
		workingDirectory?: string | null;
	}

	let { hasVisionModality = false, workingDirectory = null }: Props = $props();

	const noop = () => {};

	setChatFormActionsContext({
		disabled: false,
		hasAudioModality: false,
		hasVideoModality: false,
		get hasVisionModality() {
			return hasVisionModality;
		},
		onFilesSelect: noop,
		onFileUpload: noop,
		onSystemPromptClick: noop,
		onWorkingDirectoryClick: noop,
		get workingDirectory() {
			return workingDirectory;
		},
		get workingDirectoryLabel() {
			return workingDirectory ?? ATTACHMENT_MENU_TEXT.WORKING_DIRECTORY;
		}
	});
</script>

<div class="p-4">
	<ChatFormActionAddDrawer>
		{#snippet trigger({ disabled, onclick })}
			<Button
				aria-label={ATTACHMENT_MENU_TEXT.ADD_TO_CHAT}
				{disabled}
				{onclick}
				size="icon"
				variant="secondary"
			>
				<Plus />
			</Button>
		{/snippet}
	</ChatFormActionAddDrawer>
</div>
