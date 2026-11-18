// highlightCode and its helpers, from ui/apps/chat's utils/code.ts (only what SyntaxHighlightedCode uses)
import { CODE_BLOCK } from "../constants";
import type { HLJSApi } from "highlight.js";
import { createSubscriber } from "svelte/reactivity";

let hljs: HLJSApi | undefined;
let loading: Promise<void> | undefined;
const loadListeners = new Set<() => void>();

// lets a $derived or $effect that called highlightCode before highlight.js loaded run again once it has
const subscribeToLoad = createSubscriber((update) => {
  loadListeners.add(update);

  return () => loadListeners.delete(update);
});

/** Whether highlight.js has loaded, so highlightCode returns highlighted HTML rather than escaped plain text. */
export function isHighlighterLoaded(): boolean {
  return hljs !== undefined;
}

/**
 * Loads highlight.js and its languages (once). highlightCode starts this itself; await it where highlighted output
 * is needed right away, as in tests or a server render.
 */
export function loadHighlighter(): Promise<void> {
  loading ??= import("./highlighter").then(
    (module) => {
      hljs = module.createHighlighter();

      for (const update of [...loadListeners]) update();
    },
    (error) => {
      // let a later call try again
      loading = undefined;

      throw error;
    },
  );

  return loading;
}

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

/**
 * Highlights code with highlight.js in the given language, auto-detecting one it doesn't know.
 *
 * highlight.js loads on the first call: until it has, this returns the code escaped as plain text, and a Svelte
 * $derived or $effect that called it runs again once it has loaded. Await loadHighlighter() first to skip that.
 */
export function highlightCode(
  code: string,
  language: string,
  autoDetect = true,
): string {
  if (!code) return "";

  if (!hljs) {
    subscribeToLoad();
    loadHighlighter().catch(() => {});

    return escapeCode(trimCodePadding(code));
  }

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
