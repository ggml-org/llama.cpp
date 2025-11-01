//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'
import * as tweb from './toolweb.mjs'


let gToolsWorker = new Worker('./toolsworker.mjs', { type: 'module' });
/**
 * Maintain currently available tool/function calls
 * @type {Object<string,Object<string,any>>}
 */
export let tc_switch = {}


export async function init() {
    /**
     * @type {string[]}
     */
    let toolNames = []
    await tjs.init(gToolsWorker).then(()=>{
        for (const key in tjs.tc_switch) {
            tc_switch[key] = tjs.tc_switch[key]
            toolNames.push(key)
        }
    })
    let tNs = await tweb.init(gToolsWorker)
    for (const key in tNs) {
        tc_switch[key] = tNs[key]
        toolNames.push(key)
    }
    return toolNames
}


export function meta() {
    let tools = []
    for (const key in tc_switch) {
        tools.push(tc_switch[key]["meta"])
    }
    return tools
}


/**
 * Setup the callback that will be called when ever message
 * is recieved from the Tools Web Worker.
 * @param {(chatId: string, toolCallId: string, name: string, data: string) => void} cb
 */
export function setup(cb) {
    gToolsWorker.onmessage = function (ev) {
        cb(ev.data.cid, ev.data.tcid, ev.data.name, ev.data.data)
    }
}


/**
 * Try call the specified tool/function call.
 * Returns undefined, if the call was placed successfully
 * Else some appropriate error message will be returned.
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {string} toolargs
 */
export async function tool_call(chatid, toolcallid, toolname, toolargs) {
    for (const fn in tc_switch) {
        if (fn == toolname) {
            try {
                tc_switch[fn]["handler"](chatid, toolcallid, fn, JSON.parse(toolargs))
                return undefined
            } catch (/** @type {any} */error) {
                return `Tool/Function call raised an exception:${error.name}:${error.message}`
            }
        }
    }
    return `Unknown Tool/Function Call:${toolname}`
}
