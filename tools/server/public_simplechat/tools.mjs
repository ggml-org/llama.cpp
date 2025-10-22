//@ts-check
// DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'


let gToolsWorker = new Worker('./toolsworker.mjs', { type: 'module' });
/**
 * @type {Object<string,Object<string,any>>}
 */
export let tc_switch = {}

export async function init() {
    return tjs.init(gToolsWorker).then(()=>{
        let toolNames = []
        for (const key in tjs.tc_switch) {
            tc_switch[key] = tjs.tc_switch[key]
            toolNames.push(key)
        }
        return toolNames
    })
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
 * @param {(id: string, name: string, data: string) => void} cb
 */
export function setup(cb) {
    gToolsWorker.onmessage = function (ev) {
        cb(ev.data.id, ev.data.name, ev.data.data)
    }
}


/**
 * Try call the specified tool/function call.
 * Returns undefined, if the call was placed successfully
 * Else some appropriate error message will be returned.
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {string} toolargs
 */
export async function tool_call(toolcallid, toolname, toolargs) {
    for (const fn in tc_switch) {
        if (fn == toolname) {
            try {
                tc_switch[fn]["handler"](toolcallid, fn, JSON.parse(toolargs))
                return undefined
            } catch (/** @type {any} */error) {
                return `Tool/Function call raised an exception:${error.name}:${error.message}`
            }
        }
    }
    return `Unknown Tool/Function Call:${toolname}`
}
