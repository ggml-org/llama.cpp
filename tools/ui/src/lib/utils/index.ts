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
export { ApiError, apiFetch, apiFetchWithParams, apiPost } from './api-fetch';
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
	splitGluedClosingCodeFences,
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
export {
	createMessageCountMap,
	getMessageCount,
	buildConversationTree,
	type ConversationTreeItem
} from './conversation-utils';

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
export { parseSseJsonStream } from './sse';

// Stream session identity (conversation-id based)
export { streamIdentity } from './stream-identity';

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

// Working-directory display helpers (HOME-style tilde abbreviation)
export {
	abbreviateWorkingDir,
	abbreviateHome,
	lastPathSegment,
	formatCwdMessage,
	parseCwdMessage,
	CWD_CHANGED_PREFIX,
	CWD_CLEARED_TEXT,
	type CwdMessageInfo
} from './path-display';

// Working-directory picker search helpers
export {
	splitPathQuery,
	buildCaseInsensitiveGlob,
	buildGlobSearchArgs,
	rankEntries,
	joinPath,
	highlightMatch,
	type PathQuery
} from './working-directory';

// Shared `file_glob_search` runner with a short-lived result cache
export { runGlobSearch, runGlobSearchWithChildren } from './glob-search';

// Mention-token detection (for the `@`-triggered file/folder mention picker)
export {
	findMentionToken,
	takeMentionDismissSnapshot,
	type MentionDismissSnapshot
} from './mention-token';

// Slash-command token detection (for the `/`-triggered command picker)
export {
	findCommandToken,
	takeCommandDismissSnapshot,
	type CommandDismissSnapshot
} from './command-token';

// Tokenization for the chat-form contenteditable (mention links + code spans <-> chip DOM)
export {
	tokenizeContent,
	containsCodeSpan,
	isOffsetInCodeBlock,
	domMatchesTokens,
	syncCodeBlockHatches,
	stripBlockBoundaryLineBreaks,
	serializeContent,
	buildFragment,
	rangeToTextOffset,
	textOffsetToRange,
	badgeAwareWordJump,
	leadingBadgeEdgeOffset
} from './contenteditable-tokenizer';

// Source-space undo/redo history for the chat-form contenteditable
export { SourceHistory, type SourceHistoryEntry } from './source-history';

// Mention-badge visual contract (used by the contenteditable / rehype
// DOM paths that build the same chip without a Svelte mount)
export {
	containsFileMentionLink,
	fileMentionLinkRe,
	encodeFileLinkPath,
	decodeFileLinkPath,
	MENTION_BADGE_CLASSNAME,
	MENTION_BADGE_ICON_CLASSNAME,
	MENTION_BADGE_SVG_ATTRIBUTES,
	MENTION_BADGE_FILE_ICON_PATHS,
	MENTION_BADGE_FOLDER_ICON_PATHS,
	getMentionBadgeIconPaths,
	getMentionBadgeLabel,
	splitMentionSegments,
	buildMentionInsertion
} from './mention-badge';

// Chat template utilities
export {
	detectThinkingSupport,
	detectThinkingSupportWithReason
} from './chat-template-thinking-detector';

// Agentic content utilities (structured section derivation)
export {
	deriveAgenticSections,
	buildAssistantRawOutput,
	parseToolResultWithMedia,
	splitSearchSummaryList,
	hasAgenticContent,
	classifyToolResult,
	classifyContinueIntent
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
	isWebSearchToolName
} from './search-results';

// Cache utilities
export { TTLCache, ReactiveTTLMap } from './cache-ttl';

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
export { getBuiltinToolUi } from './built-in-tools';

// Chat command picker

export { getChatCommands } from './chat-commands';

// Sandbox tool definition
// SANDBOX_TOOL_DEFINITION is deprecated; kept for backward compatibility.
export { buildSandboxToolDefinition, SANDBOX_TOOL_DEFINITION } from './sandbox-tool';

// Cryptography utilities

export { uuid } from './uuid';

// CSS utilities
export { remToPx } from './css';

// Audio format helper (used by agentic store and chat service)
export { getAudioInputFormat } from './audio-format';
