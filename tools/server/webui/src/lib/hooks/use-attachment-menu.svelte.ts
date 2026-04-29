import { page } from '$app/state';
import { AttachmentAction } from '$lib/enums';

export interface AttachmentModalityFlags {
	hasVisionModality: boolean;
	hasAudioModality: boolean;
	hasMcpPromptsSupport: boolean;
	hasMcpResourcesSupport: boolean;
}

export interface AttachmentActionCallbacks {
	onFileUpload?: () => void;
	onSystemPromptClick?: () => void;
	onMcpPromptClick?: () => void;
	onMcpResourcesClick?: () => void;
}

export interface UseAttachmentMenuReturn {
	readonly callbacks: Record<string, () => void>;
	isItemEnabled(enabledWhen: string | undefined): boolean;
	isItemVisible(visibleWhen: string | undefined): boolean;
	getSystemMessageTooltip(): string;
}

/**
 * useAttachmentMenu - Shared logic for attachment menu components.
 *
 * Encapsulates the modality-flag checks and callback wrapping that is
 * identical across the desktop dropdown (`ChatFormActionAttachmentsDropdown`)
 * and the mobile sheet (`ChatFormActionAttachmentsSheet`).
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
			[AttachmentAction.SYSTEM_PROMPT_CLICK]: wrap(cbs.onSystemPromptClick),
			[AttachmentAction.MCP_PROMPT_CLICK]: wrap(cbs.onMcpPromptClick),
			[AttachmentAction.MCP_RESOURCES_CLICK]: wrap(cbs.onMcpResourcesClick)
		};
	});

	function isItemEnabled(enabledWhen: string | undefined): boolean {
		if (!enabledWhen || enabledWhen === 'always') return true;
		return !!modalityFlags[enabledWhen as keyof AttachmentModalityFlags];
	}

	function isItemVisible(visibleWhen: string | undefined): boolean {
		if (!visibleWhen) return true;
		return !!modalityFlags[visibleWhen as keyof AttachmentModalityFlags];
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
		isItemEnabled,
		isItemVisible,
		getSystemMessageTooltip
	};
}
