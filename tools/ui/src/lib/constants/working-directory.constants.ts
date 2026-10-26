/**
 * Constants for the working-directory picker's glob search.
 *
 * The picker glob-matches home-relative names client-side. Character classes
 * are built case-insensitively and the reserved glob metacharacters are
 * escaped (passed through literally) so a query never changes matching.
 */

/** Label shown for the working-directory picker / `/cwd` slash command. */
export const SET_WORKING_DIRECTORY_LABEL = 'Set working directory';

export const GLOB = {
	/** `C:`, the drive part of a Windows absolute path. */
	DRIVE_PREFIX_REGEX: /^[A-Za-z]:/,
	/** `C:` or `C:/`, the root of a Windows drive-absolute path. */
	DRIVE_ROOT_REGEX: /^[A-Za-z]:\/?/,
	/** Character that ends a glob character-class fragment. */
	RANGE_CLOSE: ']',
	/** Character that starts a glob character-class fragment. */
	RANGE_OPEN: '[',
	/** Query characters that carry glob meaning and are passed through literally. */
	SPECIAL_CHARS: '*?[]',
	/** `//host/share` or `//host/share/`, the root of a UNC path. */
	UNC_ROOT_REGEX: /^\/\/[^/]+\/[^/]+\/?/,
	/** Wildcard character in a glob pattern. */
	WILDCARD: '*',
	/** Separator Windows accepts alongside `/`, and a legal POSIX filename character. */
	WINDOWS_SEPARATOR: '\\'
} as const;

export const SEARCH = {
	// Search tuning for the picker's file_glob_search calls.
	DEBOUNCE_MS: 180,
	LIMIT: 100,
	// Home-relative globs descend deeper than path navigation, which only
	// needs the direct children of the parent.
	MAX_DEPTH: 6,
	MAX_RESULTS_SHOWN: 20,
	NATIVE_LIMIT: 20,
	// Native folder-picker resolution searches a shallow, bounded window.
	NATIVE_MAX_DEPTH: 4,
	PATH_NAV_MAX_DEPTH: 1
} as const;

export const FILE_GLOB_SEARCH_PICKERS = {
	/** Depth the pickers fall back to when the user setting is invalid. */
	DEFAULT_SEARCH_DEPTH: 10,
	/** Upper bound the mention search depth setting accepts. The server itself imposes no depth cap (0 = unlimited); this is a UI sanity bound. */
	MAX_SEARCH_DEPTH: 32
} as const;
