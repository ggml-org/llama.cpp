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
 * - Integrates with DialogChatAttachmentPreview for full-size viewing
 * - Validates vision modality support via `activeModelId` prop
 *
 * **Features:**
 * - Horizontal scroll with smooth navigation arrows
 * - Image thumbnails with lazy loading and error fallback
 * - File type icons for non-image files (PDF, text, audio, etc.)
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
export { default as ChatAttachmentsList } from './ChatAttachments/ChatAttachmentsList.svelte';

/**
 * Thumbnail for non-image file attachments. Displays file type icon based on extension,
 * file name (truncated), and file size.
 * Handles text files, PDFs, audio, and other document types.
 */
export { default as ChatAttachmentThumbnailFile } from './ChatAttachments/ChatAttachmentThumbnailFile.svelte';

/**
 * Thumbnail for image attachments with lazy loading and error fallback.
 * Displays image preview with configurable dimensions. Falls back to placeholder
 * on load error.
 */
export { default as ChatAttachmentThumbnailImage } from './ChatAttachments/ChatAttachmentThumbnailImage.svelte';

/**
 * Grid view of all attachments for "View All" dialog. Displays all attachments
 * in a responsive grid layout when there are too many to show inline.
 * Triggered by "+X more" button in ChatAttachmentsList.
 */
export { default as ChatAttachmentsViewAll } from './ChatAttachments/ChatAttachmentsViewAll.svelte';

/**
 * Full-size preview dialog for attachments. Opens when clicking on any attachment
 * thumbnail. Shows the attachment in full size with options to download or close.
 * Handles both image and non-image attachments with appropriate rendering.
 */
export { default as ChatAttachmentPreview } from './ChatAttachments/ChatAttachmentPreview.svelte';
/**
 *
 * FORM
 *
 * Components for the chat input area. The form handles user input, file attachments,
 * audio recording. It integrates with multiple stores:
 * - `chatStore` for message submission and generation control
 * - `modelsStore` for model selection and validation
 *
 * The form exposes a public API for programmatic control from parent components
 * (focus, height reset, model selector, validation).
 *
 */

/**
 * **ChatForm** - Main chat input component with rich features
 *
 * The primary input interface for composing and sending chat messages.
 * Orchestrates text input, file attachments, audio recording.
 * Used by ChatScreenForm and ChatMessageEditForm for both new conversations and message editing.
 *
 * **Architecture:**
 * - Composes ChatFormTextarea, ChatFormActions, and ChatFormPromptPicker
 * - Manages file upload state via `uploadedFiles` bindable prop
 * - Integrates with ModelsSelector for model selection in router mode
 * - Communicates with parent via callbacks (onSubmit, onFilesAdd, onStop, etc.)
 *
 * **Input Handling:**
 * - IME-safe Enter key handling (waits for composition end)
 * - Shift+Enter for newline, Enter for submit
 * - Paste handler for files and long text (> {pasteLongTextToFileLen} chars → file conversion)
 *
 * **Features:**
 * - Auto-resizing textarea with placeholder
 * - File upload via button dropdown (images/text/PDF), drag-drop, or paste
 * - Audio recording with WAV conversion (when model supports audio)
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
 * Dropdown button for file attachment selection. Opens a menu with options for
 * Images, Text Files, and PDF Files. Each option filters the file picker to
 * appropriate types. Images option is disabled when model lacks vision modality.
 */
export { default as ChatFormActionAttachmentsDropdown } from './ChatForm/ChatFormActions/ChatFormActionAttachmentsDropdown.svelte';

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
 * Hidden file input element for programmatic file selection.
 */
export { default as ChatFormFileInputInvisible } from './ChatForm/ChatFormFileInputInvisible.svelte';

/**
 * Helper text display below chat.
 */
export { default as ChatFormHelperText } from './ChatForm/ChatFormHelperText.svelte';

/**
 * Auto-resizing textarea with IME composition support. Automatically adjusts
 * height based on content. Handles IME input correctly (waits for composition
 * end before processing Enter key). Exposes focus() and resetHeight() methods.
 */
export { default as ChatFormTextarea } from './ChatForm/ChatFormTextarea.svelte';

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
 * - Shows attachments via ChatAttachmentsList
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
export { default as ChatMessage } from './ChatMessages/ChatMessage.svelte';

/**
 * Action buttons toolbar for messages. Displays copy, edit, delete, and regenerate
 * buttons based on message role. Includes branching controls when message has siblings.
 * Shows delete confirmation dialog with cascade delete count. Handles raw output toggle
 * for assistant messages.
 */
export { default as ChatMessageActions } from './ChatMessages/ChatMessageActions.svelte';

/**
 * Navigation controls for message siblings (conversation branches). Displays
 * prev/next arrows with current position counter (e.g., "2/5"). Enables users
 * to navigate between alternative versions of a message created by editing
 * or regenerating. Uses `conversationsStore.navigateToSibling()` for navigation.
 */
export { default as ChatMessageBranchingControls } from './ChatMessages/ChatMessageBranchingControls.svelte';

/**
 * Statistics display for assistant messages. Shows token counts (prompt/completion),
 * generation timing, tokens per second, and model name (when enabled in settings).
 * Data sourced from message.timings stored during generation.
 */
export { default as ChatMessageStatistics } from './ChatMessages/ChatMessageStatistics.svelte';

/**
 * System message display component. Renders system messages with distinct styling.
 * Visibility controlled by `showSystemMessage` config setting.
 */
export { default as ChatMessageSystem } from './ChatMessages/ChatMessageSystem.svelte';

/**
 * User message display component. Renders user messages with right-aligned bubble styling.
 * Shows message content, attachments via ChatAttachmentsList.
 * Supports inline editing mode with ChatMessageEditForm integration.
 */
export { default as ChatMessageUser } from './ChatMessages/ChatMessageUser.svelte';

/**
 * Assistant message display component. Renders assistant responses with left-aligned styling.
 * Supports both plain markdown content (via MarkdownContent) and agentic content with tool calls
 * (via ChatMessageAgenticContent). Shows model info badge, statistics, and action buttons.
 * Handles streaming state with real-time content updates.
 */
export { default as ChatMessageAssistant } from './ChatMessages/ChatMessageAssistant.svelte';

/**
 * Inline message editing form. Provides textarea for editing message content with
 * attachment management. Shows save/cancel buttons and optional "Save only" button
 * for editing without regenerating responses. Used within ChatMessage components
 * when user enters edit mode.
 */
export { default as ChatMessageEditForm } from './ChatMessages/ChatMessageEditForm.svelte';

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
 * - Composes ChatMessages, ChatScreenForm, ChatScreenHeader, and dialogs
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
 * <ChatScreen showCenteredEmpty={true} />
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
 * Header bar for chat screen. Displays conversation title (or "New Chat"),
 * model selector (in router mode), and action buttons (delete conversation).
 * Sticky positioned at the top of the chat area.
 */
export { default as ChatScreenHeader } from './ChatScreen/ChatScreenHeader.svelte';

/**
 * Processing info display during generation. Shows real-time statistics:
 * tokens per second, prompt/completion token counts, and elapsed time.
 * Data sourced from slotsService polling during active generation.
 * Only visible when `isCurrentConversationLoading` is true.
 */
export { default as ChatScreenProcessingInfo } from './ChatScreen/ChatScreenProcessingInfo.svelte';

/**
 *
 * SETTINGS
 *
 * Application settings components. Settings are persisted to localStorage via
 * the config store and synchronized with server `/props` endpoint for sampling
 * parameters. The settings panel uses a tabbed interface with mobile-responsive
 * horizontal scrolling tabs.
 *
 * **Parameter Sync System:**
 * Sampling parameters (temperature, top_p, etc.) can come from three sources:
 * 1. **Server Props**: Default values from `/props` endpoint
 * 2. **User Custom**: Values explicitly set by user (overrides server)
 * 3. **App Default**: Fallback when server props unavailable
 *
 * The `ChatSettingsParameterSourceIndicator` badge shows which source is active.
 *
 */

/**
 * **ChatSettings** - Application settings panel
 *
 * Comprehensive settings interface with categorized sections. Manages all
 * user preferences and sampling parameters. Integrates with config store
 * for persistence and ParameterSyncService for server synchronization.
 *
 * **Architecture:**
 * - Uses tabbed navigation with category sections
 * - Maintains local form state, commits on save
 * - Tracks user overrides vs server defaults for sampling params
 * - Exposes reset() method for dialog close without save
 *
 * **Categories:**
 * - **General**: API key, system message, show system messages toggle
 * - **Display**: Theme selection, message actions visibility, model info badge
 * - **Sampling**: Temperature, top_p, top_k, min_p, repeat_penalty, etc.
 * - **Penalties**: Frequency penalty, presence penalty, repeat last N
 * - **Import/Export**: Conversation backup and restore
 * - **Developer**: Debug options, disable auto-scroll
 *
 * **Parameter Sync:**
 * - Fetches defaults from server `/props` endpoint
 * - Shows source indicator badge (Custom/Server Props/Default)
 * - Real-time badge updates as user types
 * - Tracks which parameters user has explicitly overridden
 *
 * **Features:**
 * - Mobile-responsive layout with horizontal scrolling tabs
 * - Form validation with error messages
 * - Secure API key storage (masked input)
 * - Import/export conversations as JSON
 * - Reset to defaults option per parameter
 *
 * **Exported API:**
 * - `reset()` - Reset form fields to currently saved values (for cancel action)
 *
 * @example
 * ```svelte
 * <ChatSettings
 *   bind:this={settingsRef}
 *   onSave={() => dialogOpen = false}
 *   onCancel={() => { settingsRef.reset(); dialogOpen = false; }}
 * />
 * ```
 */
export { default as ChatSettings } from './ChatSettings/ChatSettings.svelte';

/**
 * Footer with save/cancel buttons for settings panel. Positioned at bottom
 * of settings dialog. Save button commits form state to config store,
 * cancel button triggers reset and close.
 */
export { default as ChatSettingsFooter } from './ChatSettings/ChatSettingsFooter.svelte';

/**
 * Form fields renderer for individual settings. Generates appropriate input
 * components based on field type (text, number, select, checkbox, textarea).
 * Handles validation, help text display, and parameter source indicators.
 */
export { default as ChatSettingsFields } from './ChatSettings/ChatSettingsFields.svelte';

/**
 * Import/export tab content for conversation data management. Provides buttons
 * to export all conversations as JSON file and import from JSON file.
 * Handles file download/upload and data validation.
 */
export { default as ChatSettingsImportExportTab } from './ChatSettings/ChatSettingsImportExportTab.svelte';

/**
 * Badge indicating parameter source for sampling settings. Shows one of:
 * - **Custom**: User has explicitly set this value (orange badge)
 * - **Server Props**: Using default from `/props` endpoint (blue badge)
 * - **Default**: Using app default, server props unavailable (gray badge)
 * Updates in real-time as user types to show immediate feedback.
 */
export { default as ChatSettingsParameterSourceIndicator } from './ChatSettings/ChatSettingsParameterSourceIndicator.svelte';

/**
 *
 * SIDEBAR
 *
 * The sidebar integrates with ShadCN's sidebar component system
 * for consistent styling and mobile responsiveness.
 * Conversations are loaded from conversationsStore and displayed in reverse
 * chronological order (most recent first).
 *
 */

/**
 * **ChatSidebar** - Chat Sidebar with actions menu and conversation list
 *
 * Collapsible sidebar displaying conversation history with search and
 * management actions. Integrates with ShadCN sidebar component for
 * consistent styling and mobile responsiveness.
 *
 * **Architecture:**
 * - Uses ShadCN Sidebar.* components for structure
 * - Fetches conversations from conversationsStore
 * - Manages search state and filtered results locally
 * - Handles conversation CRUD operations via conversationsStore
 *
 * **Navigation:**
 * - Click conversation to navigate to `/chat/[id]`
 * - New chat button navigates to `/` (root)
 * - Active conversation highlighted based on route params
 *
 * **Conversation Management:**
 * - Right-click or menu button for context menu
 * - Rename: Opens inline edit dialog
 * - Delete: Shows confirmation with conversation preview
 * - Delete All: Removes all conversations with confirmation
 *
 * **Features:**
 * - Search/filter conversations by title
 * - Conversation list with message previews (first message truncated)
 * - Active conversation highlighting
 * - Mobile-responsive collapse/expand via ShadCN sidebar
 * - New chat button in header
 * - Settings button opens DialogChatSettings
 *
 * **Exported API:**
 * - `handleMobileSidebarItemClick()` - Close sidebar on mobile after item selection
 * - `activateSearchMode()` - Focus search input programmatically
 * - `editActiveConversation()` - Open rename dialog for current conversation
 *
 * @example
 * ```svelte
 * <ChatSidebar bind:this={sidebarRef} />
 * ```
 */
export { default as ChatSidebar } from './ChatSidebar/ChatSidebar.svelte';

/**
 * Action buttons for sidebar header. Contains new chat button, settings button,
 * and delete all conversations button. Manages dialog states for settings and
 * delete confirmation.
 */
export { default as ChatSidebarActions } from './ChatSidebar/ChatSidebarActions.svelte';

/**
 * Single conversation item in sidebar. Displays conversation title (truncated),
 * last message preview, and timestamp. Shows context menu on right-click with
 * rename and delete options. Highlights when active (matches current route).
 * Handles click to navigate and keyboard accessibility.
 */
export { default as ChatSidebarConversationItem } from './ChatSidebar/ChatSidebarConversationItem.svelte';

/**
 * Search input for filtering conversations in sidebar. Filters conversation
 * list by title as user types. Shows clear button when query is not empty.
 * Integrated into sidebar header with proper styling.
 */
export { default as ChatSidebarSearch } from './ChatSidebar/ChatSidebarSearch.svelte';
