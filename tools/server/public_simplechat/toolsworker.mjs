//@ts-check
// STILL DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
// Helpers to handle tools/functions calling using web worker
// by Humans for All
//

/**
 * Expects to get a message with identifier name and code to run
 * Posts message with identifier name and data captured from console.log outputs
 */


import * as tconsole from "./toolsconsole.mjs"


self.onmessage = function (ev) {
    tconsole.console_redir()
    try {
        eval(ev.data.code)
    } catch (/** @type {any} */error) {
        console.log(`\n\nTool/Function call "${ev.data.name}" raised an exception:${error.name}:${error.message}\n\n`)
    }
    tconsole.console_revert()
    self.postMessage({ id: ev.data.id, name: ev.data.name, data: tconsole.gConsoleStr})
}
