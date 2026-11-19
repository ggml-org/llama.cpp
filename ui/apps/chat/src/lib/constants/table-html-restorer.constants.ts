/**
 * Matches <br>, <br/>, <br /> tags (case-insensitive).
 * Used to detect line breaks in table cell text content.
 */
export const BR_PATTERN = /<br\s*\/?\s*>/gi;

/**
 * Matches a complete <ul>...</ul> block.
 * Captures the inner content (group 1) for further <li> extraction.
 * Case-insensitive, allows multiline content.
 */
export const LIST_PATTERN = /^<ul>([\s\S]*)<\/ul>$/i;

/**
 * Matches individual <li>...</li> elements within a list.
 * Captures the inner content (group 1) of each list item.
 * Non-greedy to handle multiple consecutive items.
 * Case-insensitive, allows multiline content.
 */
export const LI_PATTERN = /<li>([\s\S]*?)<\/li>/gi;
