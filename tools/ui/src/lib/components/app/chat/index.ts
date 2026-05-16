/**
 *
 * ATTACHMENTS
 *
 * Components for displaying and managing different attachment types in chat messages.
 * Supports two operational modes:
 * - **Readonly mode**: For displaying stored attachments in sent messages (DatabaseMessageExtra[])
 * - **Editable mode**: For managing pending uploads in the input form (ChatUploadedFile[])
 *
 * The attachment system uses `getAttachmentDisplayItems()` utility to normalize both
 * data sources into a unified display format, enabling consistent rendering regardless
 * of the attachment origin.
 *
 */

/**
 * **ChatAttachmentsList** - Unified display for file attachments in chat
 *
 * Central component for rendering file attachments in both ChatMessage (readonly)
 * and ChatForm (editable) contexts.
 *
 * **Architecture:**
 * - Delegates rendering to specialized thumbnail components based on attachment type
 * - Manages scroll state and navigation arrows for horizontal overflow
 * - Integrates with DialogChatAttachmentsPreview for full-size gallery/single viewing
 * - Validates vision modality support via `activeModelId` prop
 *
 * **Features:**
 * - Horizontal scroll with smooth navigation arrows
 * - Image thumbnails with lazy loading and error fallback
 * - File type icons for non-image files (PDF, text, audio, etc.)
 * - MCP prompt attachments with expandable content preview
 * - Click-to-preview with full-size dialog and download option
 * - "View All" button when `limitToSingleRow` is enabled and content overflows
 * - Vision modality validation to warn about unsupported image uploads
 * - Customizable thumbnail dimensions via `imageHeight`/`imageWidth` props
 *
 * @example
 * ```svelte
 * <!-- Readonly mode (in ChatMessage) -->
 * <ChatAttachmentsList attachments={message.extra} readonly />
 *
 * <!-- Editable mode (in ChatForm) -->
 * <ChatAttachmentsList
 *   bind:uploadedFiles
 *   onFileRemove={(id) => removeFile(id)}
 *   limitToSingleRow
 *   activeModelId={selectedModel}
 * />
 * ```
 */
export { default as ChatAttachmentsList } from './ChatAttachments/ChatAttachmentsList/ChatAttachmentsList.svelte';

/**
 * Renders a single attachment item based on its type (image, file, MCP prompt, or MCP resource).
 * Delegates to specialized sub-components: ChatAttachmentsListItemThumbnailImage, ChatAttachmentsListItemThumbnailFile,
 * ChatAttachmentsListItemMcpPrompt, or ChatAttachmentsListItemMcpResource.
 */
export { default as ChatAttachmentsListItem } from './ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItem.svelte';

/**
 * Displays MCP Prompt attachment with expandable content preview.
 * Shows server name, prompt name, and allows expanding to view full prompt arguments
 * and content. Used when user selects a prompt from ChatFormPickerMcpPrompts.
 */
export { default as ChatAttachmentsListItemMcpPrompt } from './ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemMcpPrompt.svelte';

/**
 * Displays a single MCP Resource attachment with icon, name, and server info.
 * Shows loading/error states and supports remove action.
 * Used within ChatAttachmentMcpResources for individual resource display.
 */
export { default as ChatAttachmentsListItemMcpResource } from './ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemMcpResource.svelte';

/**
 * Thumbnail for non-image file attachments. Displays file type icon based on extension,
 * file name (truncated), and file size.
 * Handles text files, PDFs, audio, and other document types.
 */
export { default as ChatAttachmentsListItemThumbnailFile } from './ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemThumbnailFile.svelte';

/**
 * Thumbnail for image attachments with lazy loading and error fallback.
 * Displays image preview with configurable dimensions. Falls back to placeholder
 * on load error.
 */
export { default as ChatAttachmentsListItemThumbnailImage } from './ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemThumbnailImage.svelte';

/**
 * Unified attachment preview component for dialog display. Shows a single file
 * preview without carousel, or a gallery/carousel view when multiple items exist.
 * Uses ChatAttachmentPreviewSingle internally for each item's content.
 */
export { default as ChatAttachmentsPreview } from './ChatAttachments/ChatAttachmentsPreview.svelte';
export { default as ChatAttachmentsPreviewNavButtons } from './ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreviewNavButtons.svelte';
export { default as ChatAttachmentsPreviewFileInfo } from './ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreviewFileInfo.svelte';
export { default as ChatAttachmentsPreviewThumbnailStrip } from './ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreviewThumbnailStrip.svelte';
export { default as ChatAttachmentsPreviewCurrentItem } from './ChatAttachments/ChatAttachmentsPreview/ChatAttachmentsPreviewCurrentItem/ChatAttachmentsPreviewCurrentItem.svelte';

/**
 *
 * FORM
 *
 * Components for the chat input area. The form handles user input, file attachments,
 * audio recording, and MCP prompts & resources selection. It integrates with multiple stores:
 * - `chatStore` for message submission and generation control
 * - `modelsStore` for model selection and validation
 * - `mcpStore` for MCP prompt browsing and loading
 *
 * The form exposes a public API for programmatic control from parent components
 * (focus, height reset, model selector, validation).
 *
 */

/**
 * **ChatForm** - Main chat input component with rich features
 *
 * The primary input interface for composing and sending chat messages.
 * Orchestrates text input, file attachments, audio recording, and MCP prompts.
 * Used by ChatScreenForm and ChatMessageEditForm for both new conversations and message editing.
 *
 * **Architecture:**
 * - Composes ChatFormTextarea, ChatFormActions, and ChatFormPickerMcpPrompts
 * - Manages file upload state via `uploadedFiles` bindable prop
 * - Integrates with ModelsSelectorDropdown for model selection in router mode
 * - Communicates with parent via callbacks (onSubmit, onFilesAdd, onStop, etc.)
 *
 * **Input Handling:**
 * - IME-safe Enter key handling (waits for composition end)
 * - Shift+Enter for newline, Enter for submit
 * - Paste handler for files and long text (> {pasteLongTextToFileLen} chars → file conversion)
 * - Keyboard shortcut `/` triggers MCP prompt picker
 *
 * **Features:**
 * - Auto-resizing textarea with placeholder
 * - File upload via button dropdown (images/text/PDF), drag-drop, or paste
 * - Audio recording with WAV conversion (when model supports audio)
 * - MCP prompt picker with search and argument forms
 * - MCP reource picker with component to list attached resources at the bottom of Chat Form
 * - Model selector integration (router mode)
 * - Loading state with stop button, disabled state for errors
 *
 * **Exported API:**
 * - `focus()` - Focus the textarea programmatically
 * - `resetTextareaHeight()` - Reset textarea to default height after submit
 * - `openModelSelector()` - Open model selection dropdown
 * - `checkModelSelected(): boolean` - Validate model selection, show error if none
 *
 * @example
 * ```svelte
 * <ChatForm
 *   bind:this={chatFormRef}
 *   bind:value={message}
 *   bind:uploadedFiles
 *   {isLoading}
 *   onSubmit={handleSubmit}
 *   onFilesAdd={processFiles}
 *   onStop={handleStop}
 * />
 * ```
 */
export { default as ChatForm } from './ChatForm/ChatForm.svelte';

/**
 * Wrapper component for the "add to chat" button (Plus icon).
 * Exposes a `button` snippet that can be used inside DropdownMenu.Trigger (desktop)
 * or Sheet.Root (mobile) to maintain consistent styling while allowing
 * platform-specific trigger wrappers.
 */
export { default as ChatFormActionsAdd } from './ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionsAdd.svelte';

/**
 * Audio recording button with real-time recording indicator. Records audio
 * and converts to WAV format for upload. Only visible when the active model
 * supports audio modality and setting for automatic audio input is enabled. Shows recording duration while active.
 */
export { default as ChatFormActionRecord } from './ChatForm/ChatFormActions/ChatFormActionRecord.svelte';

/**
 * Container for chat form action buttons. Arranges file attachment, audio record,
 * and submit/stop buttons in a horizontal layout. Handles conditional visibility
 * based on model capabilities and loading state.
 */
export { default as ChatFormActions } from './ChatForm/ChatFormActions/ChatFormActions.svelte';

/**
 * Submit/stop button with loading state. Shows send icon normally, transforms
 * to stop icon during generation. Disabled when input is empty or form is disabled.
 * Triggers onSubmit or onStop callbacks based on current state.
 */
export { default as ChatFormActionSubmit } from './ChatForm/ChatFormActions/ChatFormActionSubmit.svelte';

/**
 * Model selector component for the chat form action bar. Renders either a dropdown
 * (desktop) or bottom sheet (mobile) for selecting the conversation model in router mode.
 * Exposes an `open` method for programmatically opening the selector.
 */
export { default as ChatFormActionModels } from './ChatForm/ChatFormActions/ChatFormActionModels.svelte';

/**
 * Dropdown submenu for managing tool permissions in the chat form.
 *
 * Displays a collapsible list of available tools organized by group (Built-in / JSON Schema).
 * Each group can be expanded to show individual tools with checkboxes for enabling/disabling.
 * Provides bulk enable/disable controls per group and shows enabled/total tool counts.
 * Opens the tools panel on the server when the menu opens.
 *
 * Features:
 * - Grouped tools with collapsible sections
 * - Group favicon display (MCP server icons)
 * - Per-group and per-tool toggle checkboxes
 * - Loading/error states for tool discovery
 * - Integration with toolsPanel for state management
 *
 * @example
 * ```svelte
 * <ChatFormActionAddToolsSubmenu />
 * ```
 */
export { default as ChatFormActionAddToolsSubmenu } from './ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionAddToolsSubmenu.svelte';

/**
 * Dropdown submenu for managing MCP servers in the chat form.
 *
 * Displays a searchable list of enabled MCP servers with toggle switches
 * to enable/disable each server for chat. Shows server favicon, health status,
 * and a "Manage MCP Servers" settings link.
 *
 * Features:
 * - Search/filter servers by name or URL
 * - Per-server toggle to enable/disable for chat
 * - Health check indicator (shows "Error" badge for failed servers)
 * - Server favicon display
 * - Settings link to manage MCP server configuration
 *
 * @example
 * ```svelte
 * <ChatFormActionAddMcpServersSubmenu onMcpSettingsClick={handleMcpSettingsClick} />
 * ```
 */
export { default as ChatFormActionAddMcpServersSubmenu } from './ChatForm/ChatFormActions/ChatFormActionAdd/ChatFormActionAddMcpServersSubmenu.svelte';

/**
 * Hidden file input element for programmatic file selection.
 */
export { default as ChatFormFileInputInvisible } from './ChatForm/ChatFormFileInputInvisible.svelte';

/**
 * Displays MCP Resource attachments as a horizontal carousel.
 * Shows resource name, URI, and allows clicking to view resource content.
 */
export { default as ChatFormMcpResourcesList } from './ChatForm/ChatFormMcpResourcesList.svelte';

/**
 * Auto-resizing textarea with IME composition support. Automatically adjusts
 * height based on content. Handles IME input correctly (waits for composition
 * end before processing Enter key). Exposes focus() and resetHeight() methods.
 */
export { default as ChatFormTextarea } from './ChatForm/ChatFormTextarea.svelte';

/**
 * **ChatFormPickerMcpPrompts** - MCP prompt selection interface
 *
 * Floating picker for browsing and selecting MCP Server Prompts.
 * Triggered by typing `/` in the chat input or choosing `MCP Prompt` option in ChatFormActionAddDropdown.
 * Loads prompts from connected MCP servers and allows users to select and configure them.
 *
 * **Architecture:**
 * - Fetches available prompts from mcpStore
 * - Manages selection state and keyboard navigation internally
 * - Delegates argument input to ChatFormPromptPickerArgumentForm
 * - Communicates prompt loading lifecycle via callbacks
 *
 * **Prompt Loading Flow:**
 * 1. User selects prompt → `onPromptLoadStart` called with placeholder ID
 * 2. Prompt content fetched from MCP server asynchronously
 * 3. On success → `onPromptLoadComplete` with full prompt data
 * 4. On failure → `onPromptLoadError` with error details
 *
 * **Features:**
 * - Search/filter prompts by name across all connected servers
 * - Keyboard navigation (↑/↓ to navigate, Enter to select, Esc to close)
 * - Argument input forms for prompts with required parameters
 * - Autocomplete suggestions for argument values
 * - Loading states with skeleton placeholders
 * - Server information header per prompt for visual identification
 *
 * **Exported API:**
 * - `handleKeydown(event): boolean` - Process keyboard events, returns true if handled
 *
 * @example
 * ```svelte
 * <ChatFormPickerMcpPrompts
 *   bind:this={pickerRef}
 *   isOpen={showPicker}
 *   searchQuery={promptQuery}
 *   onClose={() => showPicker = false}
 *   onPromptLoadStart={(id, info) => addPlaceholder(id, info)}
 *   onPromptLoadComplete={(id, result) => replacePlaceholder(id, result)}
 *   onPromptLoadError={(id, error) => handleError(id, error)}
 * />
 * ```
 */
export { default as ChatFormPickerMcpPrompts } from './ChatForm/ChatFormPickers/ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';

/**
 * Form for entering MCP prompt arguments. Displays input fields for each
 * required argument defined by the prompt. Validates input and submits
 * when all required fields are filled. Shows argument descriptions as hints.
 */
export { default as ChatFormPromptPickerArgumentForm } from './ChatForm/ChatFormPickers/ChatFormPickerMcpPrompts/ChatFormPromptPickerArgumentForm.svelte';

/**
 * Single argument input field with autocomplete suggestions. Fetches suggestions
 * from MCP server based on argument type. Supports keyboard navigation through
 * suggestions list. Used within ChatFormPromptPickerArgumentForm.
 */
export { default as ChatFormPromptPickerArgumentInput } from './ChatForm/ChatFormPickers/ChatFormPickerMcpPrompts/ChatFormPromptPickerArgumentInput.svelte';

/**
 * Shared popover wrapper for inline picker popovers (prompts, resources).
 * Provides consistent positioning, styling, and open/close behavior.
 */
export { default as ChatFormPickerPopover } from './ChatForm/ChatFormPickers/ChatFormPicker/ChatFormPickerPopover.svelte';

/**
 * Generic scrollable list for picker popovers. Provides search input,
 * scroll-into-view for keyboard navigation, loading skeletons, empty state,
 * and optional footer. Uses Svelte 5 snippets for item/skeleton/footer rendering.
 * Shared by ChatFormPickerMcpPrompts and ChatFormPickerMcpResources.
 */
export { default as ChatFormPickerList } from './ChatForm/ChatFormPickers/ChatFormPicker/ChatFormPickerList.svelte';

/**
 * Generic button wrapper for picker list items. Provides consistent styling,
 * hover/selected states, and data-picker-index attribute for scroll-into-view.
 * Shared by ChatFormPickerMcpPrompts and ChatFormPickerMcpResources.
 */
export { default as ChatFormPickerListItem } from './ChatForm/ChatFormPickers/ChatFormPicker/ChatFormPickerListItem.svelte';

/**
 * Generic header for picker items displaying server favicon, label, item title,
 * and optional description. Accepts `titleExtra` and `subtitle` snippets for
 * custom content like badges or URIs. Shared by both pickers.
 */
export { default as ChatFormPickerItemHeader } from './ChatForm/ChatFormPickers/ChatFormPicker/ChatFormPickerItemHeader.svelte';

/**
 * Generic skeleton loading placeholder for picker list items. Configurable
 * title width and optional badge skeleton. Shared by both pickers.
 */
export { default as ChatFormPickerListItemSkeleton } from './ChatForm/ChatFormPickers/ChatFormPicker/ChatFormPickerListItemSkeleton.svelte';

/**
 * **ChatFormPickerMcpResources** - MCP resource selection interface
 *
 * Floating picker for browsing and attaching MCP Server Resources.
 * Triggered by typing `@` in the chat input.
 * Loads resources from connected MCP servers and allows users to attach them to the chat context.
 *
 * **Features:**
 * - Search/filter resources by name, title, description, or URI across all connected servers
 * - Keyboard navigation (↑/↓ to navigate, Enter to select, Esc to close)
 * - Shows attached state for already-attached resources
 * - Loading states with skeleton placeholders
 * - Server information header per resource for visual identification
 *
 * **Exported API:**
 * - `handleKeydown(event): boolean` - Process keyboard events, returns true if handled
 */
export { default as ChatFormPickerMcpResources } from './ChatForm/ChatFormPickers/ChatFormPickerMcpResources.svelte';

/**
 * **ChatFormPickers** - Chat input picker container
 *
 * Container component that hosts both MCP prompt and MCP resource pickers.
 * Manages shared state, keyboard navigation, and coordination between the two
 * picker interfaces. Used within ChatForm for `@`-triggered pickers.
 */
export { default as ChatFormPickers } from './ChatForm/ChatFormPickers/ChatFormPickers.svelte';

/**
 *
 * MESSAGES
 *
 * Components for displaying chat messages. The message system supports:
 * - **Conversation branching**: Messages can have siblings (alternative versions)
 *   created by editing or regenerating. Users can navigate between branches.
 * - **Role-based rendering**: Different layouts for user, assistant, and system messages
 * - **Streaming support**: Real-time display of assistant responses as they generate
 * - **Agentic workflows**: Special rendering for tool calls and reasoning blocks
 *
 * The branching system uses `getMessageSiblings()` utility to compute sibling info
 * for each message based on the full conversation tree stored in the database.
 *
 */

/**
 * **ChatMessages** - Message list container with branching support
 *
 * Container component that renders the list of messages in a conversation.
 * Computes sibling information for each message to enable branch navigation.
 * Integrates with conversationsStore for message operations.
 *
 * **Architecture:**
 * - Fetches all conversation messages to compute sibling relationships
 * - Filters system messages based on user config (`showSystemMessage`)
 * - Delegates rendering to ChatMessage for each message
 * - Propagates all message operations to chatStore via callbacks
 *
 * **Branching Logic:**
 * - Uses `getMessageSiblings()` to find all messages with same parent
 * - Computes `siblingInfo: { currentIndex, totalSiblings, siblingIds }`
 * - Enables navigation between alternative message versions
 *
 * **Message Operations (delegated to chatStore):**
 * - Edit with branching: Creates new message branch, preserves original
 * - Edit with replacement: Modifies message in place
 * - Regenerate: Creates new assistant response as sibling
 * - Delete: Removes message and all descendants (cascade)
 * - Continue: Appends to incomplete assistant message
 *
 * @example
 * ```svelte
 * <ChatMessages
 *   messages={activeMessages()}
 *   onUserAction={resetAutoScroll}
 * />
 * ```
 */
export { default as ChatMessages } from './ChatMessages/ChatMessages.svelte';

/**
 * **ChatMessage** - Single message display with actions
 *
 * Renders a single chat message with role-specific styling and full action
 * support. Delegates to specialized components based on message role:
 * ChatMessageUser, ChatMessageAssistant, or ChatMessageSystem.
 *
 * **Architecture:**
 * - Routes to role-specific component based on `message.type`
 * - Manages edit mode state and inline editing UI
 * - Handles action callbacks (copy, edit, delete, regenerate)
 * - Displays branching controls when message has siblings
 *
 * **User Messages:**
 * - Shows attachments via ChatAttachments
 * - Displays MCP prompts if present
 * - Edit creates new branch or preserves responses
 *
 * **Assistant Messages:**
 * - Renders content via MarkdownContent or ChatMessageAgenticContent
 * - Shows model info badge (when enabled)
 * - Regenerate creates sibling with optional model override
 * - Continue action for incomplete responses
 *
 * **Features:**
 * - Inline editing with file attachments support
 * - Copy formatted content to clipboard
 * - Delete with confirmation (shows cascade delete count)
 * - Branching controls for sibling navigation
 * - Statistics display (tokens, timing)
 *
 * @example
 * ```svelte
 * <ChatMessage
 *   {message}
 *   {siblingInfo}
 *   onEditWithBranching={handleEdit}
 *   onRegenerateWithBranching={handleRegenerate}
 *   onNavigateToSibling={handleNavigate}
 * />
 * ```
 */
export { default as ChatMessage } from './ChatMessages/ChatMessage/ChatMessage.svelte';

/**
 * **ChatMessageAgenticContent** - Agentic workflow output display
 *
 * Specialized renderer for assistant messages with tool calls and reasoning.
 * Derives display sections from structured message data (toolCalls, reasoningContent,
 * and child tool result messages) and renders them as interactive collapsible sections.
 *
 * **Architecture:**
 * - Uses `deriveAgenticSections()` from `$lib/utils` to build sections from structured data
 * - Renders sections as CollapsibleContentBlock components
 * - Handles streaming state for progressive content display
 * - Falls back to MarkdownContent for plain text sections
 *
 * **Execution States:**
 * - **Streaming**: Animated spinner, block expanded, auto-scroll enabled
 * - **Pending**: Waiting indicator for queued tool calls
 * - **Completed**: Static display, block collapsed by default
 *
 * **Features:**
 * - JSON arguments syntax highlighting via SyntaxHighlightedCode
 * - Tool results display with formatting
 * - Plain text sections between markers rendered as markdown
 * - Smart collapse defaults (expanded while streaming, collapsed when done)
 *
 * @example
 * ```svelte
 * <ChatMessageAgenticContent
 *   content={message.content}
 *   {message}
 *   isStreaming={isGenerating}
 * />
 * ```
 */
export { default as ChatMessageAgenticContent } from './ChatMessages/ChatMessageAgenticContent.svelte';
export { default as ChatMessageActionCardPermissionRequest } from './ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardPermissionRequest.svelte';
export { default as ChatMessageActionCard } from './ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCard.svelte';
export { default as ChatMessageActionCardContinueRequest } from './ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardContinueRequest.svelte';

/**
 * Action buttons toolbar for messages. Displays copy, edit, delete, and regenerate
 * buttons based on message role. Includes branching controls when message has siblings.
 * Shows delete confirmation dialog with cascade delete count. Handles raw output toggle
 * for assistant messages.
 */
export { default as ChatMessageActionIcons } from './ChatMessages/ChatMessageActions/ChatMessageActionIcons/ChatMessageActionIcons.svelte';

/**
 * Navigation controls for message siblings (conversation branches). Displays
 * prev/next arrows with current position counter (e.g., "2/5"). Enables users
 * to navigate between alternative versions of a message created by editing
 * or regenerating. Uses `conversationsStore.navigateToSibling()` for navigation.
 */
export { default as ChatMessageActionIconsBranchingControls } from './ChatMessages/ChatMessageActions/ChatMessageActionIcons/ChatMessageActionIconsBranchingControls.svelte';

/**
 * Statistics display for assistant messages. Shows token counts (prompt/completion),
 * generation timing, tokens per second, and model name (when enabled in settings).
 * Data sourced from message.timings stored during generation.
 */
export { default as ChatMessageStatistics } from './ChatMessages/ChatMessageStatistics/ChatMessageStatistics.svelte';
export { default as ChatMessageStatisticsBadge } from './ChatMessages/ChatMessageStatistics/ChatMessageStatisticsBadge.svelte';

/**
 * MCP prompt display in user messages. Shows when user selected an MCP prompt
 * via ChatFormPickerMcpPrompts. Displays server name, prompt name, and expandable
 * content preview. Stored in message.extra as DatabaseMessageExtraMcpPrompt.
 */
export { default as ChatMessageMcpPrompt } from './ChatMessages/ChatMessage/ChatMessageMcpPrompt/ChatMessageMcpPrompt.svelte';

/**
 * Formatted content display for MCP prompt messages. Renders the full prompt
 * content with arguments in a readable format. Used within ChatMessageMcpPrompt
 * for the expanded view.
 */
export { default as ChatMessageMcpPromptContent } from './ChatMessages/ChatMessage/ChatMessageMcpPrompt/ChatMessageMcpPromptContent.svelte';

/**
 * Assistant message display component. Renders assistant responses with left-aligned styling.
 * Supports both plain markdown content (via MarkdownContent) and agentic content with tool calls
 * (via ChatMessageAgenticContent). Shows model info badge, statistics, and action buttons.
 * Handles streaming state with real-time content updates.
 */
export { default as ChatMessageAssistant } from './ChatMessages/ChatMessage/ChatMessageAssistant/ChatMessageAssistant.svelte';

/**
 * Inline message editing form. Provides textarea for editing message content with
 * attachment management. Shows save/cancel buttons and optional "Save only" button
 * for editing without regenerating responses. Used within ChatMessage components
 * when user enters edit mode.
 */
export { default as ChatMessageEditForm } from './ChatMessages/ChatMessageEditForm.svelte';

/**
 * User message display component. Renders user messages with right-aligned bubble styling.
 * Shows message content, attachments via ChatAttachmentsList, and MCP prompts if present.
 * Supports inline editing mode with ChatMessageEditForm integration.
 */
export { default as ChatMessageUser } from './ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUser.svelte';
export { default as ChatMessageUserBubble } from './ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserBubble.svelte';
export { default as ChatMessageUserPending } from './ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserPending.svelte';

/**
 * System message display component. Renders system messages with distinct styling.
 * Visibility controlled by `showSystemMessage` config setting.
 */
export { default as ChatMessageSystem } from './ChatMessages/ChatMessage/ChatMessageSystem/ChatMessageSystem.svelte';

/**
 *
 * SCREEN
 *
 * Top-level chat interface components. ChatScreen is the main container that
 * orchestrates all chat functionality. It integrates with multiple stores:
 * - `chatStore` for message operations and generation control
 * - `conversationsStore` for conversation management
 * - `serverStore` for server connection state
 * - `modelsStore` for model capabilities (vision, audio modalities)
 *
 * The screen handles the complete chat lifecycle from empty state to active
 * conversation with streaming responses.
 *
 */

/**
 * **ChatScreen** - Main chat interface container
 *
 * Top-level component that orchestrates the entire chat interface. Manages
 * messages display, input form, file handling, auto-scroll, error dialogs,
 * and server state. Used as the main content area in chat routes.
 *
 * **Architecture:**
 * - Composes ChatMessages, ChatScreenForm, and dialogs
 * - Manages auto-scroll via `createAutoScrollController()` hook
 * - Handles file upload pipeline (validation → processing → state update)
 * - Integrates with serverStore for loading/error/warning states
 * - Tracks active model for modality validation (vision, audio)
 *
 * **File Upload Pipeline:**
 * 1. Files received via drag-drop, paste, or file picker
 * 2. Validated against supported types (`isFileTypeSupported()`)
 * 3. Filtered by model modalities (`filterFilesByModalities()`)
 * 4. Empty files detected and reported via DialogEmptyFileAlert
 * 5. Valid files processed to ChatUploadedFile[] format
 * 6. Unsupported files shown in error dialog with reasons
 *
 * **State Management:**
 * - `isEmpty`: Shows centered welcome UI when no conversation active
 * - `isCurrentConversationLoading`: Tracks generation state for current chat
 * - `activeModelId`: Determines available modalities for file validation
 * - `uploadedFiles`: Pending file attachments for next message
 *
 * **Features:**
 * - Messages display with smart auto-scroll (pauses on user scroll up)
 * - File drag-drop with visual overlay indicator
 * - File validation with detailed error messages
 * - Error dialog management (chat errors, model unavailable)
 * - Server loading/error/warning states with appropriate UI
 * - Conversation deletion with confirmation dialog
 * - Processing info display (tokens/sec, timing) during generation
 * - Keyboard shortcuts (Ctrl+Shift+Backspace to delete conversation)
 *
 * @example
 * ```svelte
 * <!-- In chat route -->
 * <ChatScreen showCenteredEmpty />
 *
 * <!-- In conversation route -->
 * <ChatScreen showCenteredEmpty={false} />
 * ```
 */
export { default as ChatScreen } from './ChatScreen/ChatScreen.svelte';

/**
 * Visual overlay displayed when user drags files over the chat screen.
 * Shows drop zone indicator to guide users where to release files.
 * Integrated with ChatScreen's drag-drop file upload handling.
 */
export { default as ChatScreenDragOverlay } from './ChatScreen/ChatScreenDragOverlay.svelte';

/**
 * Chat form wrapper within ChatScreen. Positions the ChatForm component at the
 * bottom of the screen with proper padding and max-width constraints. Handles
 * the visual container styling for the input area.
 */
export { default as ChatScreenForm } from './ChatScreen/ChatScreenForm.svelte';

/**
 * Processing info display during generation. Shows real-time statistics:
 * tokens per second, prompt/completion token counts, and elapsed time.
 * Data sourced from slotsService polling during active generation.
 * Only visible when `isCurrentConversationLoading` is true.
 */
export { default as ChatScreenProcessingInfo } from './ChatScreen/ChatScreenProcessingInfo.svelte';
