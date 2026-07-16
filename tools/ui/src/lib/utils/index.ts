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
export { isTextFile, isImageFile, isPdfFile, isAudioFile, isVideoFile } from './attachment-type';

// Textarea utilities
export { default as autoResizeTextarea } from './autoresize-textarea';

// Branching utilities
export {
	filterByLeafNodeId,
	findMessageById,
	findLeafNode,
	findDescendantMessages,
	getMessageSiblings,
	buildSiblingInfoMap
} from './branching';

// Code
export {
	highlightCode,
	detectIncompleteCodeBlock,
	trimCodePadding,
	type IncompleteCodeBlock
} from './code';

// Config helpers
export { setConfigValue, getConfigValue, configToParameterRecord } from './config-helpers';

// CORS Proxy
export { buildProxiedUrl, buildProxiedHeaders } from './cors-proxy';

// URL utilities
export { extractRootDomain, sanitizeExternalUrl, canonicalizeServerUrl } from './url';

// Progress helpers
export { modelLoadFraction, modelLoadProgressText } from './progress';

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

// SSE-with-JSON stream iterator (used by built-in tool streaming, decoupled
// from chat.service.ts which embeds its own SSE parser for resume support)
export { parseSseJsonStream, type SseJsonEvent } from './sse';

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
	buildAssistantRawOutput,
	parseToolResultWithImages,
	splitSearchSummaryList,
	hasAgenticContent,
	classifyToolResult,
	type AgenticSection,
	type ToolResultLine
} from './agentic';

// Line-level unified diff for tool result rendering (`edit_file` block)
export { computeLineDiff, prefixFor, renderUnifiedDiff, type DiffLine } from './compute-line-diff';

// Partial-incremental JSON parser for streaming tool arguments
export { parsePartialJsonArgs } from './parse-partial-json-args';

// `exec_shell_command` result parsing
export { parseExecShellCommandError } from './parse-exec-shell-error';
export {
	parseExecShellCommandExitStatus,
	isExitCodeSummaryLine,
	type ExecShellExitStatus
} from './parse-exec-shell-status';

// Search-result parsing (web-search / fetch MCP tools)
export {
	SUPPORTED_WEB_SEARCH_TOOL_NAMES,
	extractSearchResults,
	extractSearchQuery,
	faviconForUrl,
	isWebSearchToolName,
	type SearchResult
} from './search-results';

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

// Tool-call meta utilities. Parsers for each built-in tool live next to
// their renderer family under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// This module only carries the helpers that genuinely cross tool
// boundaries (currently: parsing the tool-result blob into a JSON
// object).
export { tryParseToolResultObject } from './tool-call-meta';

// Per-tool UI metadata (label + icon) used by the tool-call chrome.
// Re-exported through $lib/utils so renderer components can read the
// label without depending on $lib/constants directly.
export { getBuiltinToolUi, type BuiltinToolUiEntry } from '$lib/constants/built-in-tools';

// Cryptography utilities

export { uuid } from './uuid';

// CSS utilities
export { remToPx } from './css';
