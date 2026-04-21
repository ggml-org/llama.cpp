<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { ActionIcon, ChatAttachmentsList } from '$lib/components/app';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import { Send, Edit, Trash2 } from '@lucide/svelte';
	import { getProcessingInfoContext, setMessageEditContext } from '$lib/contexts';
	import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';

	interface Props {
		class?: string;
		content: string;
		extras?: DatabaseMessageExtra[];
		onSendImmediately: () => void;
		onEdit: (newContent: string, extras?: DatabaseMessageExtra[]) => void;
		onDelete: () => void;
	}

	let {
		class: className = '',
		content,
		extras = [],
		onSendImmediately,
		onEdit,
		onDelete
	}: Props = $props();

	const processingInfoCtx = getProcessingInfoContext();
	let showProcessingInfo = $derived(processingInfoCtx.showProcessingInfo);

	let isMultiline = $state(false);
	let isEditing = $state(false);
	let editedContent = $state('');
	let editedExtras = $state<DatabaseMessageExtra[]>([]);
	let editedUploadedFiles = $state<ChatUploadedFile[]>([]);
	let messageElement: HTMLElement | undefined = $state();

	$effect(() => {
		if (!messageElement || !content.trim()) return;

		if (content.includes('\n')) {
			isMultiline = true;
			return;
		}

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const element = entry.target as HTMLElement;
				const estimatedSingleLineHeight = 24;
				isMultiline = element.offsetHeight > estimatedSingleLineHeight * 1.5;
			}
		});

		resizeObserver.observe(messageElement);

		return () => {
			resizeObserver.disconnect();
		};
	});

	function handleEdit() {
		editedContent = content;
		editedExtras = [...extras];
		editedUploadedFiles = [];
		isEditing = true;
	}

	async function handleSaveEdit() {
		const trimmed = editedContent.trim();
		if (!trimmed && editedExtras.length === 0 && editedUploadedFiles.length === 0) return;

		let finalExtras: DatabaseMessageExtra[] = $state.snapshot(editedExtras);
		if (editedUploadedFiles.length > 0) {
			const plainFiles = $state.snapshot(editedUploadedFiles);
			const result = await parseFilesToMessageExtras(plainFiles);
			const newExtras = result?.extras || [];
			finalExtras = [...finalExtras, ...newExtras];
		}

		onEdit(trimmed, finalExtras.length > 0 ? finalExtras : undefined);
		isEditing = false;
	}

	function handleCancelEdit() {
		isEditing = false;
	}

	setMessageEditContext({
		get isEditing() {
			return isEditing;
		},
		get editedContent() {
			return editedContent;
		},
		get editedExtras() {
			return editedExtras;
		},
		get editedUploadedFiles() {
			return editedUploadedFiles;
		},
		get originalContent() {
			return content;
		},
		get originalExtras() {
			return extras;
		},
		get showSaveOnlyOption() {
			return false;
		},
		setContent: (c: string) => {
			editedContent = c;
		},
		setExtras: (e: DatabaseMessageExtra[]) => {
			editedExtras = e;
		},
		setUploadedFiles: (f: ChatUploadedFile[]) => {
			editedUploadedFiles = f;
		},
		save: handleSaveEdit,
		saveOnly: handleSaveEdit,
		cancel: handleCancelEdit,
		startEdit: handleEdit
	});
</script>

<div
	use:fadeInView
	aria-label="Pending user message"
	class="group flex flex-col items-end gap-3 transition-opacity hover:opacity-80 md:gap-2 {className} sticky {showProcessingInfo
		? 'bottom-44'
		: 'bottom-32'}"
	role="group"
>
	{#if isEditing}
		<ChatMessageEditForm />
	{:else}
		{#if extras && extras.length > 0}
			<div class="mb-2 max-w-[80%]">
				<ChatAttachmentsList attachments={extras} readonly imageHeight="h-40" />
			</div>
		{/if}

		{#if content.trim()}
			<Card
				class="max-w-[80%] overflow-y-auto rounded-[1.125rem] border-none bg-primary/5 px-3.75 py-1.5 text-muted-foreground backdrop-blur-md data-[multiline]:py-2.5 dark:bg-primary/8"
				data-multiline={isMultiline ? '' : undefined}
				style="overflow-wrap: anywhere; word-break: break-word;"
			>
				<span bind:this={messageElement} class="text-md whitespace-pre-wrap">
					{content}
				</span>
			</Card>
		{/if}

		<div class="max-w-[80%]">
			<div class="relative flex h-6 items-center justify-between">
				<div class="right-0 flex items-center gap-2 opacity-100 transition-opacity">
					<div
						class="pointer-events-auto inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:opacity-100"
					>
						<ActionIcon icon={Edit} tooltip="Edit" onclick={handleEdit} />
						<ActionIcon icon={Trash2} tooltip="Delete" onclick={onDelete} />
						<ActionIcon icon={Send} tooltip="Send immediately" onclick={onSendImmediately} />
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>
