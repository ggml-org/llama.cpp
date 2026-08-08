// A general contract for tool-call meta parsers under streaming.
//
// Tool args arrive token by token, so every parser sees a long tail of partial
// JSON before it sees the real thing. Five separate UI bugs came from parsers
// deriving a display value from a half-arrived one - a basename from a partial
// path, a highlight language from a partial extension, a line range from a
// half-typed number, a whole object from a mid-stream key, and a diff from an
// `old_text` whose `new_text` had not landed yet. In every case the renderer
// could not tell "this value is empty" from "this value has not arrived".
//
// Rather than re-testing each of those by hand, this states the invariant once:
//
//   1. AVAILABILITY - once a parser yields meta, it must keep yielding meta.
//      Dropping back to null blanks the block for a frame.
//   2. SETTLEDNESS  - any value shown must already be its final value, except
//      that a string may grow toward its final value, and a field may sit at a
//      declared placeholder until it settles.
//
// A parser that satisfies both cannot flicker, because nothing it displays can
// change into something unrelated.

export interface StreamingStabilityCase {
	/** Parses one streamed prefix of the args blob into display meta. */
	parse: (argsBlob: string) => unknown;
	/** The complete args object, serialised and replayed prefix by prefix. */
	args: unknown;
	/**
	 * Values a field may legitimately hold before it settles, keyed by dotted
	 * path (array indices use `[]`). Example: a highlight language sits at
	 * `'text'` until the file extension has arrived.
	 */
	placeholders?: Record<string, unknown[]>;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Compare one observed value against the settled one, collecting violations.
 *
 * Strings may be a prefix of their final value (they are still arriving).
 * Arrays may be shorter (later elements have not arrived). Everything else must
 * already equal what it will finally be.
 */
function checkSettled(
	observed: unknown,
	settled: unknown,
	path: string,
	placeholders: Record<string, unknown[]>,
	violations: string[]
): void {
	const allowed = placeholders[path];
	if (allowed?.some((candidate) => JSON.stringify(candidate) === JSON.stringify(observed))) {
		return;
	}

	if (typeof observed === 'string' && typeof settled === 'string') {
		if (!settled.startsWith(observed)) {
			violations.push(
				`${path}: showed ${JSON.stringify(observed)}, which is not a prefix of the final ${JSON.stringify(settled)}`
			);
		}
		return;
	}

	if (Array.isArray(observed) && Array.isArray(settled)) {
		if (observed.length > settled.length) {
			violations.push(
				`${path}: showed ${observed.length} items, more than the final ${settled.length}`
			);
			return;
		}
		observed.forEach((item, index) => {
			checkSettled(item, settled[index], `${path}[]`, placeholders, violations);
		});
		return;
	}

	if (isPlainObject(observed) && isPlainObject(settled)) {
		for (const key of Object.keys(observed)) {
			checkSettled(
				observed[key],
				settled[key],
				path ? `${path}.${key}` : key,
				placeholders,
				violations
			);
		}
		return;
	}

	if (JSON.stringify(observed) !== JSON.stringify(settled)) {
		violations.push(
			`${path}: showed ${JSON.stringify(observed)} before settling on ${JSON.stringify(settled)}`
		);
	}
}

/**
 * Replay `args` one character at a time and report every way the parser's output
 * would visibly change into something other than its final value. An empty array
 * means the parser cannot flicker for this input.
 */
export function findStreamingViolations(testCase: StreamingStabilityCase): string[] {
	const argsBlob = JSON.stringify(testCase.args);
	const placeholders = testCase.placeholders ?? {};
	const settled = testCase.parse(argsBlob);

	if (settled == null) {
		return ['parser returned null for the complete args blob - the case itself is wrong'];
	}

	const violations: string[] = [];
	let becameAvailableAt = -1;

	for (let i = 1; i <= argsBlob.length; i++) {
		const observed = testCase.parse(argsBlob.slice(0, i));

		if (observed == null) {
			if (becameAvailableAt !== -1) {
				violations.push(
					`availability: went blank at prefix ${i} after first rendering at ${becameAvailableAt}`
				);
				// One report is enough; the rest would be the same gap.
				break;
			}
			continue;
		}

		if (becameAvailableAt === -1) becameAvailableAt = i;

		checkSettled(observed, settled, '', placeholders, violations);
	}

	// Collapse duplicates: the same violation repeats for every frame it spans.
	return [...new Set(violations)];
}
