/**
 * Unified exports for all utility functions
 * Import utilities from '$lib/utils' for cleaner imports
 *
 * For browser-only utilities (pdf-processing, audio-recording, svg-to-png,
 * webp-to-png, process-uploaded-files, convert-files-to-extra), use:
 * import { ... } from '$lib/utils/browser-only'
 */

// API utilities
export { getAuthHeaders, getJsonHeaders, sanitizeHeaders } from './api-headers';
export { apiFetch, apiFetchWithParams, apiPost, type ApiFetchOptions } from './api-fetch';
export { validateApiKey } from './api-key-validation';

// Attachment utilities
export { getAttachmentDisplayItems, isMcpPrompt, isMcpResource } from './attachment-display';
export { isTextFile, isImageFile, isPdfFile, isAudioFile } from './attachment-type';

// Textarea utilities
export { default as autoResizeTextarea } from './autoresize-textarea';

// Branching utilities
export {
	filterByLeafNodeId,
	findMessageById,
	findLeafNode,
	findDescendantMessages,
	getMessageSiblings,
	getMessageDisplayList,
	hasMessageSiblings,
	getNextSibling,
	getPreviousSibling
} from './branching';

// Code
export { highlightCode, detectIncompleteCodeBlock, type IncompleteCodeBlock } from './code';

// Config helpers
export { setConfigValue, getConfigValue, configToParameterRecord } from './config-helpers';

// CORS Proxy
export { buildProxiedUrl, buildProxiedHeaders } from './cors-proxy';

// URL utilities
export { extractRootDomain, sanitizeExternalUrl } from './url';

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
export { getPreviewText, generateConversationTitle } from './text';

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

// Sanitization utilities
export { sanitizeKeyValuePairKey, sanitizeKeyValuePairValue } from './sanitize';

// Image error fallback utilities
export { getImageErrorFallbackHtml } from './image-error-fallback';

// MCP utilities
export {
	detectMcpTransportFromUrl,
	parseMcpServerSettings,
	getMcpLogLevelIcon,
	getMcpLogLevelClass,
	isImageMimeType,
	parseResourcePath,
	getDisplayName,
	getResourceDisplayName,
	isCodeResource,
	isImageResource,
	getResourceIcon,
	getResourceTextContent,
	getResourceBlobContent,
	downloadResourceContent
} from './mcp';

// URI Template utilities
export {
	extractTemplateVariables,
	expandTemplate,
	isTemplateComplete,
	normalizeResourceUri,
	type UriTemplateVariable
} from './uri-template';

// Data URL utilities
export { createBase64DataUrl } from './data-url';

// Header utilities
export { parseHeadersToArray, serializeHeaders } from './headers';

// Agentic content utilities (structured section derivation)
export {
	deriveAgenticSections,
	parseToolResultWithImages,
	hasAgenticContent,
	type AgenticSection,
	type ToolResultLine
} from './agentic';

// Legacy migration utilities
export { runLegacyMigration, isMigrationNeeded } from './legacy-migration';

// Cache utilities
export { TTLCache, ReactiveTTLMap, type TTLCacheOptions } from './cache-ttl';

// Redaction utilities
export { redactValue } from './redact';

// Request inspection utilities
export {
	getRequestUrl,
	getRequestMethod,
	getRequestBody,
	summarizeRequestBody,
	formatDiagnosticErrorMessage,
	extractJsonRpcMethods,
	type RequestBodySummary
} from './request-helpers';

// Abort signal utilities
export {
	throwIfAborted,
	isAbortError,
	createLinkedController,
	createTimeoutSignal,
	withAbortSignal
} from './abort';

// Cryptography utilities

export { uuid } from './uuid';

// CSS utilities
export { remToPx } from './css';
