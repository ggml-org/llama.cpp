/**
 *
 * SERVER
 *
 * Components for displaying server connection state and handling
 * connection errors. Integrates with serverStore for state management.
 *
 */

/**
 * **ServerStatus** - Server connection status indicator
 *
 * Compact status display showing connection state, model name,
 * and context size. Used in headers and loading screens.
 *
 * **Architecture:**
 * - Reads state from serverStore (props, loading, error)
 * - Displays model name from modelsStore
 *
 * **Features:**
 * - Status dot: green (connected), yellow (connecting), red (error), gray (unknown)
 * - Status text label
 * - Model name badge with icon
 * - Context size badge
 * - Optional error action button
 *
 * @example
 * ```svelte
 * <ServerStatus showActions />
 * ```
 */
export { default as ServerStatus } from './ServerStatus.svelte';

/**
 * **ServerErrorSplash** - Full-screen connection error display
 *
 * Blocking error screen shown when server connection fails.
 * Provides retry options and API key input for authentication errors.
 *
 * **Architecture:**
 * - Detects access denied errors for API key flow
 * - Validates API key against server before saving
 * - Integrates with settingsStore for API key persistence
 *
 * **Features:**
 * - Error message display with icon
 * - Retry connection button with loading state
 * - API key input for authentication errors
 * - API key validation with success/error feedback
 * - Troubleshooting section with server start commands
 * - Animated transitions for UI elements
 *
 * @example
 * ```svelte
 * <ServerErrorSplash
 *   error={serverError}
 *   onRetry={handleRetry}
 *   showTroubleshooting
 * />
 * ```
 */
export { default as ServerErrorSplash } from './ServerErrorSplash.svelte';

/**
 * **ServerLoadingSplash** - Full-screen loading display
 *
 * Shown during initial server connection. Displays loading animation
 * with ServerStatus component for real-time connection state.
 *
 * **Features:**
 * - Animated server icon
 * - Customizable loading message
 * - Embedded ServerStatus for live updates
 *
 * @example
 * ```svelte
 * <ServerLoadingSplash message="Connecting to server..." />
 * ```
 */
export { default as ServerLoadingSplash } from './ServerLoadingSplash.svelte';
