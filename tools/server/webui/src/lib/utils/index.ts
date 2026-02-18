/**
 * Unified exports for all utility functions
 * Import utilities from '$lib/utils' for cleaner imports
 *
 * For browser-only utilities (pdf-processing, audio-recording, svg-to-png,
 * webp-to-png, process-uploaded-files, convert-files-to-extra), use:
 * import { ... } from '$lib/utils/browser-only'
 */

// API utilities
export { getAuthHeaders, getJsonHeaders } from './api-headers';
export { apiFetch, apiFetchWithParams, apiPost, type ApiFetchOptions } from './api-fetch';
export { validateApiKey } from './api-key-validation';

// Attachment utilities
export { getAttachmentDisplayItems } from './attachment-display';
export { isTextFile, isImageFile, isPdfFile, isAudioFile } from './attachment-type';

// Textarea utilities
export { default as autoResizeTextarea } from './autoresize-textarea';

// Branching utilities
export {
	filterByLeafNodeId,
	findLeafNode,
	findDescendantMessages,
	getMessageSiblings,
	getMessageDisplayList,
	hasMessageSiblings,
	getNextSibling,
	getPreviousSibling
} from './branching';

// Config helpers
export { setConfigValue, getConfigValue, configToParameterRecord } from './config-helpers';

// Conversation utilities
export { createMessageCountMap, getMessageCount } from './conversation-utils';

// Clipboard utilities
export {
	copyToClipboard,
	copyCodeToClipboard,
	formatMessageForClipboard,
	parseClipboardContent,
	hasClipboardAttachments
} from './clipboard';

// File preview utilities
export { getFileTypeLabel } from './file-preview';
export { getPreviewText } from './text';

// File type utilities
export {
	getFileTypeCategory,
	getFileTypeCategoryByExtension,
	getFileTypeByExtension,
	isFileTypeSupported
} from './file-type';

// Formatting utilities
export {
	formatFileSize,
	formatParameters,
	formatNumber,
	formatJsonPretty,
	formatTime,
	formatPerformanceTime,
	formatAttachmentText
} from './formatters';

// IME utilities
export { isIMEComposing } from './is-ime-composing';

// LaTeX utilities
export { maskInlineLaTeX, preprocessLaTeX } from './latex-protection';

// Modality file validation utilities
export {
	isFileTypeSupportedByModel,
	filterFilesByModalities,
	generateModalityErrorMessage
} from './modality-file-validation';

// Model name utilities
export { normalizeModelName, isValidModelName } from './model-names';

// Portal utilities
export { portalToBody } from './portal-to-body';

// Precision utilities
export { normalizeFloatingPoint, normalizeNumber } from './precision';

// Syntax highlighting utilities
export { getLanguageFromFilename } from './syntax-highlight-language';

// Text file utilities
export { isTextFileByName, readFileAsText, isLikelyTextFile } from './text-files';

// Debounce utilities
export { debounce } from './debounce';

// Image error fallback utilities
export { getImageErrorFallbackHtml } from './image-error-fallback';

// Data URL utilities
export { createBase64DataUrl } from './data-url';

// Cache utilities
export { TTLCache, ReactiveTTLMap, type TTLCacheOptions } from './cache-ttl';

// Abort signal utilities
export {
	throwIfAborted,
	isAbortError,
	createLinkedController,
	createTimeoutSignal,
	withAbortSignal
} from './abort';
