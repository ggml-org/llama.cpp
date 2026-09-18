/**
 * Constants for synthetic working-directory messages.
 *
 * The synthetic cwd-change message is text the UI renders as a folder row
 * and the model sees as a turn reminder. The prefix and cleared marker keep
 * the human-readable wording; the file-link regexes parse the
 * `[file:///abs/path](display)` payload back out on the UI side.
 */

import { UrlProtocol } from '$lib/enums';

export const CWD_CHANGED_PREFIX = 'Set working directory to ';
export const CWD_CLEARED_TEXT = 'Working directory cleared';

/** Trailing separator that marks a path as a directory. */
export const DIRECTORY_PATH_SUFFIX = '/';

export const HOME_TILDE = '~';
export const HOME_TILDE_PREFIX = '~/'; // tilde plus path separator

/** Scheme prefix of the file link embedded in a synthetic cwd message. */
export const FILE_URI_PREFIX = `${UrlProtocol.FILE}//`;

/** Matches the leading `[file:///abs/path](display)` link; not anchored to the end so trailing guidance may follow. */
export const CWD_LINK_REGEX = /^\[file:\/\/([\s\S]*?)\]\(([\s\S]*?)\)/;
