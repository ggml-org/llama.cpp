import { page } from '$app/state';
import { AttachmentAction } from '$lib/enums';

export interface AttachmentModalityFlags {
	hasVisionModality: boolean;
	hasAudioModality: boolean;
	hasVideoModality: boolean;
}

export interface AttachmentActionCallbacks {
	onFileUpload?: () => void;
	onSystemPromptClick?: () => void;
}

export interface UseAttachmentMenuReturn {
	readonly callbacks: Record<string, () => void>;
	isItemEnabled(enabledWhen: string | undefined): boolean;
	getSystemMessageTooltip(): string;
}

/**
 * useAttachmentMenu - Shared logic for attachment menu components.
 *
 * Encapsulates the modality-flag checks and callback wrapping that is
 * identical across the desktop dropdown (`ChatFormActionAddDropdown`)
 * and the mobile sheet (`ChatFormActionAddSheet`).
 *
 * @param getFlags   - Getter returning the current modality capability flags.
 * @param getCallbacks - Getter returning the raw action callbacks from props.
 * @param close      - Function that dismisses the hosting UI element (dropdown / sheet).
 */
export function useAttachmentMenu(
	getFlags: () => AttachmentModalityFlags,
	getCallbacks: () => AttachmentActionCallbacks,
	close: () => void
): UseAttachmentMenuReturn {
	const modalityFlags = $derived(getFlags());
	const callbacks = $derived.by(() => {
		const cbs = getCallbacks();
		const wrap = (fn?: () => void) => () => {
			close();
			fn?.();
		};

		return {
			[AttachmentAction.FILE_UPLOAD]: wrap(cbs.onFileUpload),
			[AttachmentAction.SYSTEM_PROMPT_CLICK]: wrap(cbs.onSystemPromptClick)
		};
	});

	function isItemEnabled(enabledWhen: string | undefined): boolean {
		if (!enabledWhen || enabledWhen === 'always') return true;

		return !!modalityFlags[enabledWhen as keyof AttachmentModalityFlags];
	}

	function getSystemMessageTooltip(): string {
		return !page.params.id
			? 'Add custom system message for a new conversation'
			: 'Inject custom system message at the beginning of the conversation';
	}

	return {
		get callbacks() {
			return callbacks;
		},
		getSystemMessageTooltip,
		isItemEnabled
	};
}
