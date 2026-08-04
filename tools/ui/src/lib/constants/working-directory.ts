/**
 * Constants for the working-directory picker's glob search.
 *
 * The picker glob-matches home-relative names client-side. Character classes
 * are built case-insensitively and the reserved glob metacharacters are
 * escaped (passed through literally) so a query never changes matching.
 */

export const GLOB_WILDCARD = '*';

/** Character that starts and ends a glob character-class fragment. */
export const GLOB_RANGE_OPEN = '[';
export const GLOB_RANGE_CLOSE = ']';

/** Query characters that carry glob meaning and are passed through literally. */
export const GLOB_SPECIAL_CHARS = '*?[]';

/** Separator Windows accepts alongside `/`, and a legal POSIX filename character. */
export const WINDOWS_SEPARATOR = '\\';

/** `C:`, the drive part of a Windows absolute path. */
export const DRIVE_PREFIX_REGEX = /^[A-Za-z]:/;

/** `C:` or `C:/`, the root of a Windows drive-absolute path. */
export const DRIVE_ROOT_REGEX = /^[A-Za-z]:\/?/;

/** `//host/share` or `//host/share/`, the root of a UNC path. */
export const UNC_ROOT_REGEX = /^\/\/[^/]+\/[^/]+\/?/;

// Search tuning for the picker's file_glob_search calls.
export const SEARCH_DEBOUNCE_MS = 180;
export const SEARCH_LIMIT = 100;
export const MAX_RESULTS_SHOWN = 20;
// Home-relative globs descend deeper than path navigation, which only
// needs the direct children of the parent.
export const SEARCH_MAX_DEPTH = 6;
export const PATH_NAV_MAX_DEPTH = 1;
// Native folder-picker resolution searches a shallow, bounded window.
export const NATIVE_MAX_DEPTH = 4;
export const NATIVE_LIMIT = 20;
