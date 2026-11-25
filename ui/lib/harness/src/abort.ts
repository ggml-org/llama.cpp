const ABORT_ERROR_NAME = "AbortError";
// TypeErrors a browser throws when a fetch stream is cut by page unload, navigation or a network drop
const ABORT_LIKE_MESSAGE_PATTERNS = [
  /input stream/i,
  /network connection was lost/i,
  /load failed/i,
];

export function isAbortError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;

  if (error.name === ABORT_ERROR_NAME) return true;

  return (
    error instanceof TypeError && ABORT_LIKE_MESSAGE_PATTERNS.some((re) => re.test(error.message))
  );
}
