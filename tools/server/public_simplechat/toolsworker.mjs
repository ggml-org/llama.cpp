//@ts-check
// STILL DANGER DANGER DANGER - Simple and Stupid - Using from a discardable VM better.
// Helpers to handle tools/functions calling using web worker
// by Humans for All
//

/**
 * Expects to get a message with id (session and toolcall), name and code to run
 * Posts message with id (session and toolcall), name and data captured from console.log outputs
 */


import * as tconsole from "./toolsconsole.mjs"
import * as xpromise from "./xpromise.mjs"


self.onmessage = async function (ev) {
    console.info("DBUG:WW:OnMessage started...")
    tconsole.console_redir()
    try {
        await xpromise.evalWithPromiseTracking(ev.data.code);
    } catch (/** @type {any} */error) {
        console.log(`\nTool/Function call "${ev.data.name}" raised an exception:${error.name}:${error.message}`)
    }
    tconsole.console_revert()
    self.postMessage({ cid: ev.data.cid, tcid: ev.data.tcid, name: ev.data.name, data: tconsole.gConsoleStr})
    console.info("DBUG:WW:OnMessage done")
}
