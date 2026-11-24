// Reading tool-call arguments while they stream: a lenient parse for cut-off
// JSON, and a targeted read of one string field for arguments that hold
// whole files.

const PARTIAL_JSON_CACHE_MAX_SIZE = 32;
const partialJsonCache = new Map<string, Record<string, unknown> | null>();

function asObject(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * Parse tool-call arguments that may be cut short mid-stream: an open string
 * and open containers are closed so the keys received so far can be read.
 * Null when the arguments are not an object. Memoized, since a trace parses the
 * same arguments on every streamed chunk.
 */
export function parsePartialJsonArgs(args: string): Record<string, unknown> | null {
  const cached = partialJsonCache.get(args);

  if (cached !== undefined) return cached;

  let result: Record<string, unknown> | null;

  try {
    result = asObject(JSON.parse(args));
  } catch {
    result = scanPartialJson(args);
  }

  if (partialJsonCache.size >= PARTIAL_JSON_CACHE_MAX_SIZE) {
    partialJsonCache.delete(partialJsonCache.keys().next().value!);
  }

  partialJsonCache.set(args, result);

  return result;
}

function scanPartialJson(args: string): Record<string, unknown> | null {
  let inString = false;
  let escape = false;

  const stack: ("{" | "[")[] = [];

  for (const ch of args) {
    if (escape) {
      escape = false;

      continue;
    }

    if (ch === "\\" && inString) {
      escape = true;

      continue;
    }

    if (ch === '"') {
      inString = !inString;

      continue;
    }

    if (inString) continue;

    if (ch === "{" || ch === "[") stack.push(ch);
    else if (ch === "}" || ch === "]") {
      if (stack.pop() !== (ch === "}" ? "{" : "[")) return null;
    }
  }

  let completed = args;

  // a dangling escape: keep the backslash as a literal so the string can close
  if (escape) completed += "\\";

  if (inString) completed += '"';
  else completed = completed.replace(/,?\s*$/, "");

  for (let i = stack.length - 1; i >= 0; i--) completed += stack[i] === "{" ? "}" : "]";

  try {
    return asObject(JSON.parse(completed));
  } catch {
    return null;
  }
}

const fieldPatterns = new Map<string, RegExp>();

/**
 * Read one string field out of tool-call arguments without parsing them all:
 * write and edit tools carry whole files, and a trace line needs only the
 * path. The first of `keys` that holds a string wins; undefined when none does.
 */
export function extractToolArgString(args: string, keys: readonly string[]): string | undefined {
  for (const key of keys) {
    let pattern = fieldPatterns.get(key);

    if (!pattern) {
      pattern = new RegExp(`"${key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"`);
      fieldPatterns.set(key, pattern);
    }

    const match = pattern.exec(args);

    if (!match) continue;

    try {
      const value: unknown = JSON.parse(`"${match[1]}"`);

      if (typeof value === "string") return value;
    } catch {
      // the next key, then the caller's full parse
    }
  }

  return undefined;
}

/** The file name of a path, for a trace line; undefined without a path. */
export function fileNameFromPath(path: string | undefined): string | undefined {
  if (!path) return undefined;

  return path.split(/[\\/]/).pop() || path;
}

/**
 * Memoize `fn` by a string key, holding at most `maxSize` results; a full cache
 * is emptied, which suits values that are cheap to rebuild. `keyOf` returns
 * null for a call not worth caching.
 */
export function memoizeBounded<Args extends unknown[], Result>(
  fn: (...args: Args) => Result,
  keyOf: (...args: Args) => string | null,
  maxSize: number,
): (...args: Args) => Result {
  const cache = new Map<string, Result>();

  return (...args) => {
    const key = keyOf(...args);

    if (key === null) return fn(...args);

    if (cache.has(key)) return cache.get(key) as Result;

    const result = fn(...args);

    if (cache.size >= maxSize) cache.clear();

    cache.set(key, result);

    return result;
  };
}
