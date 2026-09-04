<script lang="ts">
	import DialogConfirmation from '$lib/components/app/dialogs/DialogConfirmation.svelte';
	import { DownloadConfirmAction } from '$lib/enums';
	import { modelsStore } from '$lib/stores';

	interface Props {
		/** Action being confirmed; drives the wording. */
		action: DownloadConfirmAction;
		/** `<repo>:<tag>` the action targets. */
		repoWithTag: string;
		onClose: () => void;
		/** Overrides the default store removal; defaults to removing the entry. */
		onConfirm?: (repoWithTag: string) => void;
		open?: boolean;
	}

	let { action, onClose, onConfirm, open = true, repoWithTag }: Props = $props();

	// Both actions resolve through the same store removal (cancelDownload drops a
	// running download's partial files or a cached model's files); only the copy
	// differs. One component so the discover chips and the selector rows word the
	// destructive confirmations identically.
	const COPY = {
		[DownloadConfirmAction.CANCEL]: {
			cancelText: 'Keep downloading',
			confirmText: 'Cancel download',
			description: (name: string) =>
				`This stops the download of ${name} and removes the partial files. Pause it instead to keep the progress.`,
			title: 'Cancel download'
		},
		[DownloadConfirmAction.DELETE]: {
			cancelText: 'Keep model',
			confirmText: 'Delete',
			description: (name: string) =>
				`This permanently removes ${name} from disk. You can download it again later.`,
			title: 'Delete model'
		}
	} as const;

	let copy = $derived(COPY[action]);
	let displayName = $derived(modelsStore.toDisplayName(repoWithTag));

	function confirm() {
		if (onConfirm) onConfirm(repoWithTag);
		else void modelsStore.status.cancelDownload(repoWithTag);

		onClose();
	}
</script>

<DialogConfirmation
	cancelText={copy.cancelText}
	confirmText={copy.confirmText}
	description={copy.description(displayName)}
	onCancel={onClose}
	onConfirm={confirm}
	{open}
	title={copy.title}
	variant="destructive"
/>
