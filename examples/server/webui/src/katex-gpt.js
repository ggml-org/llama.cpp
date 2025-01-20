import katex from 'katex';

// Adapted from https://github.com/SchneeHertz/markdown-it-katex-gpt
// MIT license

const defaultOptions = {
  delimiters: [
    { left: '\\[', right: '\\]', display: true },
    { left: '\\(', right: '\\)', display: false },
  ],
};

export function renderLatexHTML(content, display = false) {
  return katex.renderToString(content, {
    throwOnError: false,
    output: 'mathml',
    displayMode: display,
  });
}

function escapedBracketRule(options) {
  return (state, silent) => {
    const max = state.posMax;
    const start = state.pos;

    for (const { left, right, display } of options.delimiters) {

      // Check if it starts with the left delimiter
      if (!state.src.slice(start).startsWith(left)) continue;

      // Skip the length of the left delimiter
      let pos = start + left.length;

      // Find the matching right delimiter
      while (pos < max) {
        if (state.src.slice(pos).startsWith(right)) {
          break;
        }
        pos++;
      }

      // No matching right delimiter found, skip to the next match
      if (pos >= max) continue;

      // If not in silent mode, convert LaTeX formula to MathML
      if (!silent) {
        const content = state.src.slice(start + left.length, pos);
        try {
          const renderedContent = renderLatexHTML(content, display);
          const token = state.push('html_inline', '', 0);
          token.content = renderedContent;
        } catch (e) {
          console.error(e);
        }
      }

      // Update position, skip the length of the right delimiter
      state.pos = pos + right.length;
      return true;
    }
  }
}

export default function (md, options = defaultOptions) {
  md.inline.ruler.after('text', 'escaped_bracket', escapedBracketRule(options));
}
