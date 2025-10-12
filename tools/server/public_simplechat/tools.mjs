//@ts-check
// DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'


/**
 * @type {Object<string,Object<string,any>>}
 */
export let tc_switch = {}

export function setup() {
    for (const key in tjs.tc_switch) {
        tc_switch[key] = tjs.tc_switch[key]
    }
}

export function meta() {
    let tools = []
    for (const key in tc_switch) {
        tools.push(tc_switch[key]["meta"])
    }
    return tools
}


/**
 * Try call the specified tool/function call and return its response
 * @param {string} toolname
 * @param {string} toolargs
 */
export async function tool_call(toolname, toolargs) {
    for (const fn in tc_switch) {
        if (fn == toolname) {
            try {
                tc_switch[fn]["handler"](fn, JSON.parse(toolargs))
                return tc_switch[fn]["result"]
            } catch (/** @type {any} */error) {
                return `Tool/Function call raised an exception:${error.name}:${error.message}`
            }
        }
    }
    return `Unknown Tool/Function Call:${toolname}`
}
