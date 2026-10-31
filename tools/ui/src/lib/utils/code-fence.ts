const CODE_FENCE_RE = /```/g;

/**
 * Is `offset` inside a fenced code block region? Toggle-based: an
 * odd number of ``` fences before the offset means the position
 * sits in block content. This also counts the still-OPEN fence
 * while the user is typing a block (no closing ``` yet), so Enter
 * can add a line instead of submitting the message.
 */
export function isOffsetInCodeBlock(source: string, offset: number): boolean {
	let inside = false;

	CODE_FENCE_RE.lastIndex = 0;

	let match: RegExpExecArray | null;

	while ((match = CODE_FENCE_RE.exec(source)) !== null) {
		if (match.index + match[0].length > offset) break;

		inside = !inside;
	}

	return inside;
}
