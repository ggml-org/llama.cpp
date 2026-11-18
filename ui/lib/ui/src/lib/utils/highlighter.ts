// highlight.js with every lowlight language registered; code.ts loads this module on first use, so an app keeps its
// ~870 KB out of its first load
import { CODE_BLOCK } from "../constants";
import type { HLJSApi } from "highlight.js";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import { all as languages } from "lowlight";

/** A highlight.js instance with every lowlight language, and a bash that also marks a command's name and flags. */
export function createHighlighter(): HLJSApi {
  const highlighter = hljs.newInstance();

  // reuse lowlight's language modules: the full `highlight.js` entry bundles a second, CommonJS copy of every language
  for (const [name, language] of Object.entries(languages))
    highlighter.registerLanguage(name, language);

  // highlight.js's bash marks strings, variables and keywords but leaves a command's name and its flags plain, so a
  // command (`curl -H … -d …`) reads as one colour. Mark them too: the name as a built-in, the flags as attributes.
  highlighter.registerLanguage("bash", (api) => {
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

  return highlighter;
}
