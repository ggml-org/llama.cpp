<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import { Send, Edit, Trash2 } from '@lucide/svelte';
	import { getProcessingInfoContext, setMessageEditContext } from '$lib/contexts';
	import { parseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';
	import ChatMessageUserBubble from './ChatMessageUserBubble.svelte';

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

	let isEditing = $state(false);
	let editedContent = $state('');
	let editedExtras = $state<DatabaseMessageExtra[]>([]);
	let editedUploadedFiles = $state<ChatUploadedFile[]>([]);

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
		<ChatMessageUserBubble
			{content}
			attachments={extras}
			textColorClass="text-muted-foreground"
			cardBgClass="dark:bg-primary/8"
			maxHeightStyle="overflow-wrap: anywhere; word-break: break-word;"
		/>

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
