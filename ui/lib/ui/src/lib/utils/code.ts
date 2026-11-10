// highlightCode and its helpers, from ui/apps/chat's utils/code.ts (only what SyntaxHighlightedCode uses)
import { CODE_BLOCK } from "$lib/constants";
import hljs from "highlight.js";
import bash from "highlight.js/lib/languages/bash";

// highlight.js's bash marks strings, variables and keywords but leaves a command's name and its flags plain, so a
// command (`curl -H … -d …`) reads as one colour. Mark them too: the name as a built-in, the flags as attributes.
hljs.registerLanguage("bash", (api) => {
  const base = bash(api);

  return {
    ...base,
    contains: [
      { match: CODE_BLOCK.SHELL_COMMAND_REGEX, scope: "built_in" },
      { match: CODE_BLOCK.SHELL_FLAG_REGEX, scope: "attr" },
      ...(base.contains ?? []),
    ],
  };
});

/**
 * Strips empty lines (whitespace-only) from the start and end of code, so hljs emits no phantom row before or after
 * the body; internal blank lines stay.
 */
export function trimCodePadding(code: string): string {
  return code
    .replace(CODE_BLOCK.TRIM_LEADING_PADDING_REGEX, "")
    .replace(CODE_BLOCK.TRIM_TRAILING_PADDING_REGEX, "");
}

function escapeCode(code: string): string {
  return code
    .replace(CODE_BLOCK.AMPERSAND_REGEX, "&amp;")
    .replace(CODE_BLOCK.LT_REGEX, "&lt;")
    .replace(CODE_BLOCK.GT_REGEX, "&gt;");
}

const HIGHLIGHT_CACHE_MAX_SIZE = 64;
const highlightCache = new Map<string, string>();

/** Highlights code with highlight.js in the given language, auto-detecting one it doesn't know. */
export function highlightCode(
  code: string,
  language: string,
  autoDetect = true,
): string {
  if (!code) return "";

  const cacheKey = `${language}:${autoDetect}:${code}`;
  const cached = highlightCache.get(cacheKey);

  if (cached) return cached;

  const trimmed = trimCodePadding(code);

  let result: string;

  try {
    const lang = language.toLowerCase();

    if (hljs.getLanguage(lang)) {
      result = hljs.highlight(trimmed, { language: lang }).value;
    } else if (autoDetect) {
      result = hljs.highlightAuto(trimmed).value;
    } else {
      result = escapeCode(trimmed);
    }
  } catch {
    result = escapeCode(trimmed);
  }

  if (highlightCache.size >= HIGHLIGHT_CACHE_MAX_SIZE) {
    highlightCache.delete(highlightCache.keys().next().value!);
  }

  highlightCache.set(cacheKey, result);

  return result;
}
