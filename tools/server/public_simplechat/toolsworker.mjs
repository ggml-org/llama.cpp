//@ts-check
// STILL DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
// Helpers to handle tools/functions calling using web worker
// by Humans for All
//

import * as tconsole from "./toolsconsole.mjs"

tconsole.console_redir()

onmessage = async (ev) => {
    try {
        eval(ev.data)
    } catch (/** @type {any} */error) {
        console.log(`\n\nTool/Function call raised an exception:${error.name}:${error.message}\n\n`)
    }
    tconsole.console_revert()
    postMessage(tconsole.gConsoleStr)
}
