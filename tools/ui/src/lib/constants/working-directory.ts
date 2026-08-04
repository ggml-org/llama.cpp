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
