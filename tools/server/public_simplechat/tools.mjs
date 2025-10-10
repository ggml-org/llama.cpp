//@ts-check
// DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'


/**
 * @type {Object<string,Object>}
 */
let tc_switch = {}

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

