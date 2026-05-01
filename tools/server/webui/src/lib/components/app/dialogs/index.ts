/**
 *
 * DIALOGS
 *
 * Modal dialog components for the chat application.
 *
 * All dialogs use ShadCN Dialog or AlertDialog components for consistent
 * styling, accessibility, and animation. They integrate with application
 * stores for state management and data access.
 *
 */

/**
 * **DialogMcpServerAddNew** - Add new MCP server dialog
 *
 * Modal dialog for adding a new MCP server with URL and optional headers.
 * Validates URL format and integrates with mcpStore and conversationsStore.
 */
export { default as DialogMcpServerAddNew } from './DialogMcpServerAddNew.svelte';

/**
 *
 * CONFIRMATION DIALOGS
 *
 * Dialogs for user action confirmations. Use AlertDialog for blocking
 * confirmations that require explicit user decision before proceeding.
 *
 */

/**
 * **DialogConfirmation** - Generic confirmation dialog
 *
 * Reusable confirmation dialog with customizable title, description,
 * and action buttons. Supports destructive action styling and custom icons.
 * Used for delete confirmations, irreversible actions, and important decisions.
 *
 * **Architecture:**
 * - Uses ShadCN AlertDialog
 * - Supports variant styling (default, destructive)
 * - Customizable button labels and callbacks
 *
 * **Features:**
 * - Customizable title and description text
 * - Destructive variant with red styling for dangerous actions
 * - Custom icon support in header
 * - Cancel and confirm button callbacks
 * - Keyboard accessible (Escape to cancel, Enter to confirm)
 *
 * @example
 * ```svelte
 * <DialogConfirmation
 *   bind:open={showDelete}
 *   title="Delete conversation?"
 *   description="This action cannot be undone."
 *   variant="destructive"
 *   onConfirm={handleDelete}
 *   onCancel={() => showDelete = false}
 * />
 * ```
 */
export { default as DialogConfirmation } from './DialogConfirmation.svelte';

/**
 * **DialogConversationTitleUpdate** - Conversation rename confirmation
 *
 * Confirmation dialog shown when editing the first user message in a conversation.
 * Asks user whether to update the conversation title to match the new message content.
 *
 * **Architecture:**
 * - Uses ShadCN AlertDialog
 * - Shows current vs proposed title comparison
 * - Triggered by ChatMessages when first message is edited
 *
 * **Features:**
 * - Side-by-side display of current and new title
 * - "Keep Current Title" and "Update Title" action buttons
 * - Styled title previews in muted background boxes
 *
 * @example
 * ```svelte
 * <DialogConversationTitleUpdate
 *   bind:open={showTitleUpdate}
 *   currentTitle={conversation.name}
 *   newTitle={truncatedMessageContent}
 *   onConfirm={updateTitle}
 *   onCancel={() => showTitleUpdate = false}
 * />
 * ```
 */
export { default as DialogConversationTitleUpdate } from './DialogConversationTitleUpdate.svelte';

/**
 *
 * CONTENT PREVIEW DIALOGS
 *
 * Dialogs for previewing and displaying content in full-screen or modal views.
 *
 */

/**
 * **DialogCodePreview** - Full-screen code/HTML preview
 *
 * Full-screen dialog for previewing HTML or code in an isolated iframe.
 * Used by MarkdownContent component for previewing rendered HTML blocks
 * from code blocks in chat messages.
 *
 * **Architecture:**
 * - Uses ShadCN Dialog with full viewport layout
 * - Sandboxed iframe execution (allow-scripts only)
 * - Clears content when closed for security
 *
 * **Features:**
 * - Full viewport iframe preview
 * - Sandboxed execution environment
 * - Close button with mix-blend-difference for visibility over any content
 * - Automatic content cleanup on close
 * - Supports HTML preview with proper isolation
 *
 * @example
 * ```svelte
 * <DialogCodePreview
 *   bind:open={showPreview}
 *   code={htmlContent}
 *   language="html"
 * />
 * ```
 */
export { default as DialogCodePreview } from './DialogCodePreview.svelte';

/**
 *
 * ATTACHMENT DIALOGS
 *
 * Dialogs for viewing and managing file attachments. Support both
 * uploaded files (pending) and stored attachments (in messages).
 *
 */

/**
 * **DialogChatAttachmentsPreview** - Unified attachment preview dialog
 *
 * Modal dialog for previewing file attachments. Automatically adapts to the
 * number of items: shows a single file preview without carousel for one item,
 * or a gallery with carousel navigation for multiple items.
 *
 * **Architecture:**
 * - Wraps ChatAttachmentsPreview component in ShadCN Dialog
 * - Accepts uploadedFiles and attachments arrays as data sources
 * - Filters out MCP prompts and MCP resources from display
 *
 * **Features:**
 * - Single item mode: direct preview without navigation controls
 * - Multi-item mode: gallery with left/right arrows and thumbnail strip
 * - File type aware preview (images, text, PDFs, audio)
 * - File name and size/count display in header
 *
 * @example
 * ```svelte
 * <!-- Gallery with focus on 2nd item -->
 * <DialogChatAttachmentsPreview
 *   bind:open={showPreview}
 *   uploadedFiles={pendingFiles}
 *   attachments={message.extra}
 *   activeModelId={currentModel}
 *   previewFocusIndex={1}
 * />
 * ```
 */
export { default as DialogChatAttachmentsPreview } from './DialogChatAttachmentsPreview.svelte';

/**
 *
 * ERROR & ALERT DIALOGS
 *
 * Dialogs for displaying errors, warnings, and alerts to users.
 * Provide context about what went wrong and recovery options.
 *
 */

/**
 * **DialogChatError** - Chat/generation error display
 *
 * Alert dialog for displaying chat and generation errors with context
 * information. Supports different error types with appropriate styling
 * and messaging.
 *
 * **Architecture:**
 * - Uses ShadCN AlertDialog for modal display
 * - Differentiates between timeout and server errors
 * - Shows context info when available (token counts)
 *
 * **Error Types:**
 * - **timeout**: TCP timeout with timer icon, red destructive styling
 * - **server**: Server error with warning icon, amber warning styling
 *
 * **Features:**
 * - Type-specific icons (TimerOff for timeout, AlertTriangle for server)
 * - Error message display in styled badge
 * - Context info showing prompt tokens and context size
 * - Close button to dismiss
 *
 * @example
 * ```svelte
 * <DialogChatError
 *   bind:open={showError}
 *   type="server"
 *   message={errorMessage}
 *   contextInfo={{ n_prompt_tokens: 1024, n_ctx: 4096 }}
 * />
 * ```
 */
export { default as DialogChatError } from './DialogChatError.svelte';

/**
 * **DialogEmptyFileAlert** - Empty file upload warning
 *
 * Alert dialog shown when user attempts to upload empty files. Lists the
 * empty files that were detected and removed from attachments, with
 * explanation of why empty files cannot be processed.
 *
 * **Architecture:**
 * - Uses ShadCN AlertDialog for modal display
 * - Receives list of empty file names from ChatScreen
 * - Triggered during file upload validation
 *
 * **Features:**
 * - FileX icon indicating file error
 * - List of empty file names in monospace font
 * - Explanation of what happened and why
 * - Single "Got it" dismiss button
 *
 * @example
 * ```svelte
 * <DialogEmptyFileAlert
 *   bind:open={showEmptyAlert}
 *   emptyFiles={['empty.txt', 'blank.md']}
 * />
 * ```
 */
export { default as DialogEmptyFileAlert } from './DialogEmptyFileAlert.svelte';

/**
 * **DialogFileUploadError** - File upload compatibility error
 *
 * Alert dialog shown when files cannot be uploaded due to type incompatibility
 * or model modality restrictions. Displays a categorized list of problematic
 * files with explanations and shows which file types the current model supports.
 *
 * **Architecture:**
 * - Uses ShadCN AlertDialog for modal display
 * - Receives structured file error data from ChatScreen
 * - Triggered during file upload validation in processFiles()
 *
 * **Features:**
 * - Categorized display: unsupported types vs modality restrictions
 * - File name in monospace with contextual error messages
 * - Summary of supported file types for the current model
 * - Scrollable content area for large error lists
 * - Single "Got it" dismiss button
 *
 * @example
 * ```svelte
 * <DialogFileUploadError
 *   bind:open={showFileError}
 *   fileErrorData={errorData}
 * />
 * ```
 */
export { default as DialogFileUploadError } from './DialogFileUploadError.svelte';

/**
 * **DialogModelNotAvailable** - Model unavailable error
 *
 * Alert dialog shown when the requested model (from URL params or selection)
 * is not available on the server. Displays the requested model name and
 * offers selection from available models.
 *
 * **Architecture:**
 * - Uses ShadCN AlertDialog for modal display
 * - Integrates with SvelteKit navigation for model switching
 * - Receives available models list from modelsStore
 *
 * **Features:**
 * - Warning icon with amber styling
 * - Requested model name display in styled badge
 * - Scrollable list of available models
 * - Click model to navigate with updated URL params
 * - Cancel button to dismiss without selection
 *
 * @example
 * ```svelte
 * <DialogModelNotAvailable
 *   bind:open={showModelError}
 *   modelName={requestedModel}
 *   availableModels={modelsList}
 * />
 * ```
 */
export { default as DialogModelNotAvailable } from './DialogModelNotAvailable.svelte';

/**
 *
 * DATA MANAGEMENT DIALOGS
 *
 * Dialogs for managing conversation data, including import/export
 * and selection operations.
 *
 */

/**
 * **DialogConversationSelection** - Conversation picker for import/export
 *
 * Dialog for selecting conversations during import or export operations.
 * Displays list of conversations with checkboxes for multi-selection.
 * Used by ChatSettingsImportExportTab for data management.
 *
 * **Architecture:**
 * - Wraps ConversationSelection component in ShadCN Dialog
 * - Supports export mode (select from local) and import mode (select from file)
 * - Resets selection state when dialog opens
 * - High z-index to appear above settings dialog
 *
 * **Features:**
 * - Multi-select with checkboxes
 * - Conversation title and message count display
 * - Select all / deselect all controls
 * - Mode-specific descriptions (export vs import)
 * - Cancel and confirm callbacks with selected conversations
 *
 * @example
 * ```svelte
 * <DialogConversationSelection
 *   bind:open={showExportSelection}
 *   conversations={allConversations}
 *   messageCountMap={messageCounts}
 *   mode="export"
 *   onConfirm={handleExport}
 *   onCancel={() => showExportSelection = false}
 * />
 * ```
 */
export { default as DialogConversationSelection } from './DialogConversationSelection.svelte';

/**
 *
 * MODEL INFORMATION DIALOGS
 *
 * Dialogs for displaying model and server information.
 *
 */

/**
 * **DialogModelInformation** - Model details display
 *
 * Dialog showing comprehensive information about the currently loaded model
 * and server configuration. Displays model metadata, capabilities, and
 * server settings in a structured table format.
 *
 * **Architecture:**
 * - Uses ShadCN Dialog with wide layout for table display
 * - Fetches data from serverStore (props) and modelsStore (metadata)
 * - Auto-fetches models when dialog opens if not loaded
 *
 * **Information Displayed:**
 * - **Model**: Name with copy button
 * - **File Path**: Full path to model file with copy button
 * - **Context Size**: Current context window size
 * - **Training Context**: Original training context (if available)
 * - **Model Size**: File size in human-readable format
 * - **Parameters**: Parameter count (e.g., "7B", "70B")
 * - **Embedding Size**: Embedding dimension
 * - **Vocabulary Size**: Token vocabulary size
 * - **Vocabulary Type**: Tokenizer type (BPE, etc.)
 * - **Parallel Slots**: Number of concurrent request slots
 * - **Modalities**: Supported input types (text, vision, audio)
 * - **Build Info**: Server build information
 * - **Chat Template**: Full Jinja template in scrollable code block
 *
 * **Features:**
 * - Copy buttons for model name and path
 * - Modality badges with icons
 * - Responsive table layout with container queries
 * - Loading state while fetching model info
 * - Scrollable chat template display
 *
 * @example
 * ```svelte
 * <DialogModelInformation bind:open={showModelInfo} />
 * ```
 */
export { default as DialogModelInformation } from './DialogModelInformation.svelte';

/**
 * **DialogMcpResourcesBrowser** - MCP resources browser dialog
 *
 * Dialog for browsing and attaching MCP resources to chat context.
 * Displays resources from connected MCP servers in a tree structure
 * with preview panel and multi-select support.
 *
 * **Architecture:**
 * - Uses ShadCN Dialog with two-panel layout
 * - Left panel: McpResourcesBrowser with tree navigation
 * - Right panel: McpResourcePreview for selected resource
 * - Integrates with mcpStore for resource fetching and attachment
 *
 * **Features:**
 * - Tree-based resource navigation by server and path
 * - Single and multi-select with shift+click
 * - Resource preview with content display
 * - Quick attach button per resource
 * - Batch attach for multiple selections
 *
 * @example
 * ```svelte
 * <DialogMcpResourcesBrowser
 *   bind:open={showResources}
 *   onAttach={handleResourceAttach}
 * />
 * ```
 */
export { default as DialogMcpResourcesBrowser } from './DialogMcpResourcesBrowser.svelte';

/**
 * **DialogMcpResourcePreview** - MCP resource content preview
 *
 * Dialog for previewing the content of a stored MCP resource attachment.
 * Displays the resource content with syntax highlighting for code,
 * image rendering for images, and plain text for other content.
 *
 * **Features:**
 * - Syntax highlighted code preview
 * - Image rendering for image resources
 * - Copy to clipboard and download actions
 * - Server name and favicon display
 * - MIME type badge
 *
 * @example
 * ```svelte
 * <DialogMcpResourcePreview
 *   bind:open={previewOpen}
 *   extra={mcpResourceExtra}
 * />
 * ```
 */
export { default as DialogMcpResourcePreview } from './DialogMcpResourcePreview.svelte';
