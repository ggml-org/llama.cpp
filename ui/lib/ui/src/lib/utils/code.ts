// highlightCode and its helpers, from ui/apps/chat's utils/code.ts (only what SyntaxHighlightedCode uses)
import { CODE_BLOCK } from "../constants";
import {
  HIGHLIGHT_AUTO_DETECT_LANGUAGES,
  HIGHLIGHT_LANGUAGE_ALIASES,
  HIGHLIGHT_LANGUAGE_LOADERS,
  HIGHLIGHT_SUB_LANGUAGES,
} from "./highlight-languages";
import type { HLJSApi, LanguageFn } from "highlight.js";
import { createSubscriber } from "svelte/reactivity";

// highlight.js and each language load on first use, every language its own chunk, so an app's first load carries none
let hljs: HLJSApi | undefined;
let coreLoading: Promise<HLJSApi> | undefined;
const languageLoading = new Map<string, Promise<void>>();
/** Languages loaded together with every language they embed, so highlightCode can use them. */
const readyLanguages = new Set<string>();
/** Each loaded language's definition as highlight.js ships it, without this module's bash changes. */
const languageDefinitions = new Map<string, LanguageFn>();
const loadListeners = new Set<() => void>();

// lets a $derived or $effect that called highlightCode before its language loaded run again once it has
const subscribeToLoad = createSubscriber((update) => {
  loadListeners.add(update);

  return () => loadListeners.delete(update);
});

// highlight.js's bash marks strings, variables and keywords but leaves a command's name and its flags plain, so a
// command (`curl -H … -d …`) reads as one colour. Mark them too: the name as a built-in, the flags as attributes.
function withShellCommands(bash: LanguageFn): LanguageFn {
  return (api) => {
    const base = bash(api);

    return {
      ...base,
      contains: [
        { match: CODE_BLOCK.SHELL_COMMAND_REGEX, scope: "built_in" },
        { match: CODE_BLOCK.SHELL_FLAG_REGEX, scope: "attr" },
        ...(base.contains ?? []),
      ],
    };
  };
}

/** The highlight.js name of a language or one of its aliases (`ts` -> `typescript`), or undefined for one it lacks. */
export function resolveHighlightLanguage(language: string): string | undefined {
  const key = language.toLowerCase();

  if (Object.hasOwn(HIGHLIGHT_LANGUAGE_LOADERS, key)) return key;

  return Object.hasOwn(HIGHLIGHT_LANGUAGE_ALIASES, key)
    ? HIGHLIGHT_LANGUAGE_ALIASES[key]
    : undefined;
}

/** Whether a language (or alias) has loaded, so highlightCode highlights it rather than returning plain text. */
export function isHighlightLanguageLoaded(language: string): boolean {
  const name = resolveHighlightLanguage(language);

  return name !== undefined && readyLanguages.has(name);
}

/** The definitions of every loaded language, as highlight.js ships them, to register with another highlighter. */
export function getLoadedHighlightLanguages(): Record<string, LanguageFn> {
  return Object.fromEntries(
    [...readyLanguages].map((name) => [name, languageDefinitions.get(name)!]),
  );
}

function loadCore(): Promise<HLJSApi> {
  coreLoading ??= import("highlight.js/lib/core").then(
    (module) => (hljs = module.default.newInstance()),
    (error) => {
      coreLoading = undefined;

      throw error;
    },
  );

  return coreLoading;
}

function loadLanguage(name: string): Promise<void> {
  let loading = languageLoading.get(name);

  if (!loading) {
    loading = Promise.all([
      loadCore(),
      HIGHLIGHT_LANGUAGE_LOADERS[name](),
    ]).then(([core, module]) => {
      languageDefinitions.set(name, module.default);
      core.registerLanguage(
        name,
        name === CODE_BLOCK.SHELL_LANGUAGE
          ? withShellCommands(module.default)
          : module.default,
      );
    });
    // let a later call try again
    loading.catch(() => languageLoading.delete(name));
    languageLoading.set(name, loading);
  }

  return loading;
}

/**
 * Loads languages (names or aliases; unknown ones are skipped) with every language they embed, or every language
 * when called without any. highlightCode loads what it needs itself; await this where highlighted output is needed
 * right away, as in tests or a server render.
 */
export async function loadHighlightLanguages(
  languages?: readonly string[],
): Promise<void> {
  const names = new Set<string>();
  const queue = languages
    ? languages
        .map(resolveHighlightLanguage)
        .filter((name) => name !== undefined)
    : Object.keys(HIGHLIGHT_LANGUAGE_LOADERS);

  while (queue.length) {
    const name = queue.pop()!;

    if (names.has(name)) continue;

    names.add(name);
    queue.push(...(HIGHLIGHT_SUB_LANGUAGES[name] ?? []));
  }

  const missing = [...names].filter((name) => !readyLanguages.has(name));

  if (!missing.length) return;

  await Promise.all(missing.map(loadLanguage));

  for (const name of missing) readyLanguages.add(name);
  for (const update of [...loadListeners]) update();
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
 * Highlights code with highlight.js in the given language (a name or an alias). For a language highlight.js lacks it
 * auto-detects one among lowlight's common languages, unless autoDetect is false.
 *
 * Each language loads on first use: until it has, this returns the code escaped as plain text, and a Svelte $derived
 * or $effect that called it runs again once it has loaded. Await loadHighlightLanguages() first to skip that.
 */
export function highlightCode(
  code: string,
  language: string,
  autoDetect = true,
): string {
  if (!code) return "";

  const name = resolveHighlightLanguage(language);
  const needed = name
    ? [name]
    : autoDetect
      ? HIGHLIGHT_AUTO_DETECT_LANGUAGES
      : [];
  const trimmed = trimCodePadding(code);

  if (!needed.length) return escapeCode(trimmed);

  if (!hljs || !needed.every((n) => readyLanguages.has(n))) {
    subscribeToLoad();
    loadHighlightLanguages(needed).catch(() => {});

    return escapeCode(trimmed);
  }

  const cacheKey = `${name ?? ""}:${code}`;
  const cached = highlightCache.get(cacheKey);

  if (cached) return cached;

  let result: string;

  try {
    result = name
      ? hljs.highlight(trimmed, { language: name }).value
      : hljs.highlightAuto(trimmed, [...HIGHLIGHT_AUTO_DETECT_LANGUAGES]).value;
  } catch {
    result = escapeCode(trimmed);
  }

  if (highlightCache.size >= HIGHLIGHT_CACHE_MAX_SIZE) {
    highlightCache.delete(highlightCache.keys().next().value!);
  }

  highlightCache.set(cacheKey, result);

  return result;
}
