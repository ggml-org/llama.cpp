<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import { ArrowUp, Edit, Trash2 } from '@lucide/svelte';
	import { getProcessingInfoContext } from '$lib/contexts';
	import { useMessageEditContext } from '$lib/hooks/use-message-edit-context.svelte';
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

	const editCtx = useMessageEditContext({
		getContent: () => content,
		getExtras: () => extras,
		onSave: (content, extras) => onEdit(content, extras)
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
	{#if editCtx.isEditing}
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
						<ActionIcon icon={Edit} tooltip="Edit" onclick={editCtx.handleEdit} />
						<ActionIcon icon={Trash2} tooltip="Delete" onclick={onDelete} />
						<ActionIcon icon={ArrowUp} tooltip="Send immediately" onclick={onSendImmediately} />
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>
