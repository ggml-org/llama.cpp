//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'
import * as tweb from './toolweb.mjs'
import * as tdb from './tooldb.mjs'
import * as mChatMagic from './simplechat.js'


/**
 * Maintain currently available tool/function calls
 * @type {Object<string,Object<string,any>>}
 */
export let tc_switch = {}


/**
 * @param {mChatMagic.Me} me
 */
function setup_workers(me) {
    me.workers.js =  new Worker('./toolsworker.mjs', { type: 'module' });
    me.workers.db = new Worker('./toolsdbworker.mjs', { type: 'module' });
}


/**
 * @param {mChatMagic.Me} me
 */
export async function init(me) {
    setup_workers(me);
    /**
     * @type {string[]}
     */
    let toolNames = []
    await tjs.init(me).then(()=>{
        for (const key in tjs.tc_switch) {
            tc_switch[key] = tjs.tc_switch[key]
            toolNames.push(key)
        }
    })
    await tdb.init(me).then(()=>{
        for (const key in tdb.tc_switch) {
            tc_switch[key] = tdb.tc_switch[key]
            toolNames.push(key)
        }
    })
    let tNs = await tweb.init(me)
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
