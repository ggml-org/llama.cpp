<script lang="ts">
	import { Download, Pin, PinOff, Trash2, X } from '@lucide/svelte';
	import { ActionIcon, DialogConfirmation } from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { TooltipSide } from '$lib/enums';

	interface Props {
		class?: string;
		selectedCount: number;
		visibleCount: number;
		allVisibleSelected: boolean;
		someVisibleSelected: boolean;
		someSelectedPinned: boolean;
		pinStateIsMixed: boolean;
		onSelectAllToggle: () => void;
		onBulkPinToggle: () => void;
		onBulkExport: () => void;
		onBulkDelete: () => void;
		onClose: () => void;
	}

	let {
		class: className = '',
		selectedCount,
		visibleCount,
		allVisibleSelected,
		someVisibleSelected,
		someSelectedPinned,
		pinStateIsMixed,
		onSelectAllToggle,
		onBulkPinToggle,
		onBulkExport,
		onBulkDelete,
		onClose
	}: Props = $props();

	let showDeleteDialog = $state(false);

	function handleDeleteClick() {
		showDeleteDialog = true;
	}

	function handleDeleteConfirm() {
		showDeleteDialog = false;
		onBulkDelete();
	}

	function handleDeleteCancel() {
		showDeleteDialog = false;
	}

	const hasSelection = $derived(selectedCount > 0);
	const isMasterChecked = $derived(allVisibleSelected);
	const isMasterIndeterminate = $derived(!allVisibleSelected && someVisibleSelected);

	const pinTooltip = $derived(
		hasSelection
			? pinStateIsMixed
				? 'Unavailable for mixed state selection'
				: someSelectedPinned
					? selectedCount === 1
						? 'Unpin'
						: 'Unpin all'
					: selectedCount === 1
						? 'Pin'
						: 'Pin all'
			: 'Pin'
	);

	const pinDisabled = $derived(!hasSelection || pinStateIsMixed);
</script>

<div
	role="toolbar"
	aria-label="Bulk actions for selected conversations"
	class="flex items-center gap-1.5 rounded-xl border border-border/50 bg-background/50 px-2 py-1.5 shadow-sm backdrop-blur-xl {className}"
>
	<label class="flex min-w-0 cursor-pointer items-center gap-2">
		<Checkbox
			checked={isMasterChecked}
			indeterminate={isMasterIndeterminate}
			onCheckedChange={onSelectAllToggle}
			aria-label={isMasterChecked ? 'Deselect all' : 'Select all'}
		/>

		<span class="truncate text-xs font-medium text-muted-foreground">
			{selectedCount} / {visibleCount} selected
		</span>
	</label>

	<div class="ml-auto flex items-center gap-0.75">
		<ActionIcon
			icon={someSelectedPinned ? PinOff : Pin}
			tooltip={pinTooltip}
			tooltipSide={TooltipSide.TOP}
			disabled={pinDisabled}
			ariaLabel={pinTooltip}
			size="sm"
			iconSize="h-3.5 w-3.5"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {pinDisabled
				? 'cursor-not-allowed'
				: ''} {!pinDisabled ? 'opacity-100' : 'opacity-40'}"
			onclick={onBulkPinToggle}
		/>

		<ActionIcon
			icon={Download}
			tooltip={hasSelection ? 'Export' : 'Export'}
			tooltipSide={TooltipSide.TOP}
			disabled={!hasSelection}
			ariaLabel="Export selected"
			size="sm"
			iconSize="h-3.5 w-3.5"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {hasSelection
				? 'opacity-100'
				: 'opacity-40'}"
			onclick={onBulkExport}
		/>

		<ActionIcon
			icon={Trash2}
			tooltip="Delete selected"
			tooltipSide={TooltipSide.TOP}
			disabled={!hasSelection}
			ariaLabel="Delete selected"
			size="sm"
			iconSize="h-3.5 w-3.5 text-destructive"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-destructive/10! dark:hover:bg-destructive/20! disabled:hover:bg-transparent {hasSelection
				? 'opacity-100'
				: 'opacity-40'}"
			onclick={handleDeleteClick}
		/>

		<div class="mx-1 h-4 w-px bg-border" aria-hidden="true"></div>

		<ActionIcon
			icon={X}
			tooltip="Exit bulk selection mode"
			tooltipSide={TooltipSide.TOP}
			ariaLabel="Exit bulk selection mode"
			size="sm"
			iconSize="h-3.5 w-3.5"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent!"
			onclick={onClose}
		/>
	</div>
</div>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title="Delete {selectedCount} conversation{selectedCount === 1 ? '' : 's'}"
	description="This action cannot be undone. The selected conversation{selectedCount === 1
		? ''
		: 's'} and {selectedCount === 1
		? 'its'
		: 'their'} messages will be permanently removed, including any forks."
	confirmText={selectedCount === 1 ? 'Delete' : `Delete ${selectedCount}`}
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDeleteConfirm}
	onCancel={handleDeleteCancel}
/>
