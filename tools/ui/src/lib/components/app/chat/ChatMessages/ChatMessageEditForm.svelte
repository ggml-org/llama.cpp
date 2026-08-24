<script lang="ts">
	import { AlertTriangle, Check, X } from '@lucide/svelte';
	import { ChatForm, DialogConfirmation } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import { Textarea } from '$lib/components/ui/textarea';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { KeyboardKey, MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores';
	import { processFilesToChatUploaded } from '$lib/utils/browser-only';

	const editCtx = getChatMessageEditContext();

	let saveWithoutRegenerate = $state(false);
	let showDiscardDialog = $state(false);
	let branchAfterEdit = $state(false);

	let isUserMessage = $derived(editCtx.messageRole === MessageRole.USER);
	// An assistant turn is plain text in two fields. Attachments, file mentions, MCP
	// prompts and the model picker all belong to composing a request, so its editor is
	// a pair of textareas rather than the conversation chat form.
	let isAssistantMessage = $derived(editCtx.messageRole === MessageRole.ASSISTANT);

	let hasUnsavedChanges = $derived.by(() => {
		if (editCtx.editedContent !== editCtx.originalContent) return true;

		if (editCtx.editedReasoning !== editCtx.originalReasoning) return true;

		if (editCtx.editedUploadedFiles.length > 0) return true;

		const extrasChanged =
			editCtx.editedExtras.length !== editCtx.originalExtras.length ||
			editCtx.editedExtras.some((extra, i) => extra !== editCtx.originalExtras[i]);

		if (extrasChanged) return true;

		return false;
	});

	let hasAttachments = $derived(
		(editCtx.editedExtras && editCtx.editedExtras.length > 0) ||
			(editCtx.editedUploadedFiles && editCtx.editedUploadedFiles.length > 0)
	);

	let canSubmit = $derived(
		editCtx.editedContent.trim().length > 0 ||
			editCtx.editedReasoning.trim().length > 0 ||
			hasAttachments
	);

	function handleGlobalKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			attemptCancel();
		}
	}

	// Plain Enter inserts a newline in the assistant fields, which hold long text,
	// so the accelerator carries a modifier.
	function handleFieldKeydown(event: KeyboardEvent) {
		if (event.key !== KeyboardKey.ENTER) return;

		if (!event.ctrlKey && !event.metaKey) return;

		event.preventDefault();
		handleSubmit();
	}

	function attemptCancel() {
		if (hasUnsavedChanges) {
			showDiscardDialog = true;
		} else {
			editCtx.cancel();
		}
	}

	function handleSubmit() {
		if (!canSubmit) return;

		if (isUserMessage && saveWithoutRegenerate && editCtx.showSaveOnlyOption) {
			editCtx.saveOnly();
		} else {
			if (editCtx.showBranchAfterEditOption) {
				editCtx.setShouldBranchAfterEdit?.(branchAfterEdit);
			}

			editCtx.save();
		}

		saveWithoutRegenerate = false;
		branchAfterEdit = false;
	}

	function handleContentInput(event: Event & { currentTarget: EventTarget & HTMLTextAreaElement }) {
		editCtx.setContent(event.currentTarget.value);
	}

	function handleReasoningInput(
		event: Event & { currentTarget: EventTarget & HTMLTextAreaElement }
	) {
		editCtx.setReasoning?.(event.currentTarget.value);
	}

	function handleAttachmentRemove(index: number) {
		const newExtras = [...editCtx.editedExtras];

		newExtras.splice(index, 1);
		editCtx.setExtras(newExtras);
	}

	function handleUploadedFileRemove(fileId: string) {
		const newFiles = editCtx.editedUploadedFiles.filter((f) => f.id !== fileId);

		editCtx.setUploadedFiles(newFiles);
	}

	async function handleFilesAdd(files: File[]) {
		const processed = await processFilesToChatUploaded(files);

		editCtx.setUploadedFiles([...editCtx.editedUploadedFiles, ...processed]);
	}

	let contentEl = $state<HTMLTextAreaElement | null>(null);

	$effect(() => {
		if (!contentEl) return;

		contentEl.focus();
		contentEl.setSelectionRange(contentEl.value.length, contentEl.value.length);
	});

	$effect(() => {
		chatStore.setEditModeActive(handleFilesAdd);

		return () => {
			chatStore.clearEditMode();
		};
	});
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

{#if isAssistantMessage}
	<div class="grid w-full max-w-[80%] gap-3">
		{#if editCtx.showReasoningField}
			<div class="grid gap-1">
				<Label class="text-xs text-muted-foreground" for="edit-reasoning">Reasoning</Label>

				<Textarea
					class="max-h-64 overflow-y-auto font-mono text-xs"
					id="edit-reasoning"
					oninput={handleReasoningInput}
					onkeydown={handleFieldKeydown}
					value={editCtx.editedReasoning}
				/>
			</div>
		{/if}

		<div class="grid gap-1">
			<Label class="text-xs text-muted-foreground" for="edit-response">Response</Label>

			<Textarea
				bind:ref={contentEl}
				class="max-h-96 overflow-y-auto"
				id="edit-response"
				oninput={handleContentInput}
				onkeydown={handleFieldKeydown}
				value={editCtx.editedContent}
			/>
		</div>
	</div>
{:else}
	<div class="relative w-full max-w-[80%]">
		<ChatForm
			bind:uploadedFiles={editCtx.editedUploadedFiles}
			attachments={editCtx.editedExtras}
			onAttachmentRemove={handleAttachmentRemove}
			onFilesAdd={handleFilesAdd}
			onSubmit={handleSubmit}
			onUploadedFileRemove={handleUploadedFileRemove}
			onValueChange={editCtx.setContent}
			placeholder="Edit your message..."
			showContextGauge={false}
			showWorkingDirectory={false}
			value={editCtx.editedContent}
		/>
	</div>
{/if}

<div class="mt-2 flex w-full max-w-[80%] items-center justify-between">
	{#if isUserMessage && editCtx.showSaveOnlyOption}
		<div class="flex items-center gap-2">
			<Switch bind:checked={saveWithoutRegenerate} class="scale-75" id="save-only-switch" />

			<label class="cursor-pointer text-xs text-muted-foreground" for="save-only-switch">
				Update without re-sending
			</label>
		</div>
	{:else if editCtx.showBranchAfterEditOption}
		<div class="flex items-center gap-2">
			<Switch bind:checked={branchAfterEdit} class="scale-75" id="branch-after-edit" />

			<label class="cursor-pointer text-xs text-muted-foreground" for="branch-after-edit">
				Branch conversation after edit
			</label>
		</div>
	{:else}
		<div></div>
	{/if}

	<div class="flex items-center gap-1">
		<Button class="h-7 px-3 text-xs" onclick={attemptCancel} size="sm" variant="ghost">
			<X class="mr-1 h-3 w-3" />

			Cancel
		</Button>

		{#if isAssistantMessage}
			<Button
				class="h-7 px-3 text-xs"
				disabled={!canSubmit}
				onclick={handleSubmit}
				size="sm"
				variant="ghost"
			>
				<Check class="mr-1 h-3 w-3" />

				Save
			</Button>
		{/if}
	</div>
</div>

<DialogConfirmation
	bind:open={showDiscardDialog}
	cancelText="Keep editing"
	confirmText="Discard"
	description="You have unsaved changes. Are you sure you want to discard them?"
	icon={AlertTriangle}
	onCancel={() => (showDiscardDialog = false)}
	onConfirm={editCtx.cancel}
	title="Discard changes?"
	variant="destructive"
/>
