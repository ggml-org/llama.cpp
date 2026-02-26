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
 *
 * SETTINGS DIALOGS
 *
 * Dialogs for application and server configuration.
 *
 */

/**
 * **DialogChatSettings** - Settings dialog wrapper
 *
 * Modal dialog containing ChatSettings component with proper
 * open/close state management and automatic form reset on open.
 *
 * **Architecture:**
 * - Wraps ChatSettings component in ShadCN Dialog
 * - Manages open/close state via bindable `open` prop
 * - Resets form state when dialog opens to discard unsaved changes
 *
 * @example
 * ```svelte
 * <DialogChatSettings bind:open={showSettings} />
 * ```
 */
export { default as DialogChatSettings } from './DialogChatSettings.svelte';

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
 * **DialogChatAttachmentPreview** - Full-size attachment preview
 *
 * Modal dialog for viewing file attachments at full size. Supports different
 * file types with appropriate preview modes: images, text files, PDFs, and audio.
 *
 * **Architecture:**
 * - Wraps ChatAttachmentPreview component in ShadCN Dialog
 * - Accepts either uploaded file or stored attachment as data source
 * - Resets preview state when dialog opens
 *
 * **Features:**
 * - Full-size image display with proper scaling
 * - Text file content with syntax highlighting
 * - PDF preview with text/image view toggle
 * - Audio file placeholder with download option
 * - File name and size display in header
 * - Download button for all file types
 * - Vision modality check for image attachments
 *
 * @example
 * ```svelte
 * <!-- Preview uploaded file -->
 * <DialogChatAttachmentPreview
 *   bind:open={showPreview}
 *   uploadedFile={selectedFile}
 *   activeModelId={currentModel}
 * />
 *
 * <!-- Preview stored attachment -->
 * <DialogChatAttachmentPreview
 *   bind:open={showPreview}
 *   attachment={selectedAttachment}
 * />
 * ```
 */
export { default as DialogChatAttachmentPreview } from './DialogChatAttachmentPreview.svelte';

/**
 * **DialogChatAttachmentsViewAll** - Grid view of all attachments
 *
 * Dialog showing all attachments in a responsive grid layout. Triggered by
 * "+X more" button in ChatAttachmentsList when there are too many attachments
 * to display inline.
 *
 * **Architecture:**
 * - Wraps ChatAttachmentsViewAll component in ShadCN Dialog
 * - Supports both readonly (message view) and editable (form) modes
 * - Displays total attachment count in header
 *
 * **Features:**
 * - Responsive grid layout for all attachments
 * - Thumbnail previews with click-to-expand
 * - Remove button in editable mode
 * - Configurable thumbnail dimensions
 * - Vision modality validation for images
 *
 * @example
 * ```svelte
 * <DialogChatAttachmentsViewAll
 *   bind:open={showAllAttachments}
 *   attachments={message.extra}
 *   readonly
 * />
 * ```
 */
export { default as DialogChatAttachmentsViewAll } from './DialogChatAttachmentsViewAll.svelte';

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
