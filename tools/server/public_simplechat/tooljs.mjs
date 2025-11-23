//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling wrt
// * javascript interpreter
// * simple arithmatic calculator
// using the js specific web worker.
// by Humans for All
//

import * as mChatMagic from './simplechat.js'
import * as mToolsMgr from './tools.mjs'


let gMe = /** @type{mChatMagic.Me} */(/** @type {unknown} */(null));


let sysdatetime_meta = {
        "type": "function",
        "function": {
            "name": "sys_date_time",
            "description": "Returns the current system date and time. The template argument helps control which parts of date and time are returned",
            "parameters": {
                "type": "object",
                "properties": {
                     "template": {
                        "type": "string",
                        "description": `Template is used to control what is included in the returned date time string.
                            It can be any combination of Y,m,d,H,M,S,w. Here
                            Y - FullYear 4 digits, m - Month 2 digits, d - Day 2 digits,
                            H - hour 2 digits 24 hours format, M - minutes 2 digits, S - seconds 2 digits,
                            w - day of week (0(sunday)..6(saturday)).
                            Any other char will be returned as is.

                            YmdTHMS is a useful date time template, which includes all the key parts.
                            Remember that the template characters are case sensitive.
                            `
                    }
               },
                "required": ["template"]
            }
        }
    }


/**
 * Implementation of the system date and time.
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function sysdatetime_run(chatid, toolcallid, toolname, obj) {
    let dt = new Date()
    let tmpl = obj['template'];
    if ((tmpl == undefined) || (tmpl == "")) {
        tmpl = 'YmdTHMS';
    }
    let sDT = ""
    for (const c of tmpl) {
        switch (c) {
            case 'Y':
                sDT += dt.getFullYear().toString().padStart(4, '0')
                break;
            case 'm':
                sDT += (dt.getMonth()+1).toString().padStart(2, '0')
                break;
            case 'd':
                sDT += dt.getDate().toString().padStart(2, '0')
                break;
            case 'H':
                sDT += dt.getHours().toString().padStart(2, '0')
                break;
            case 'M':
                sDT += dt.getMinutes().toString().padStart(2, '0')
                break;
            case 'S':
                sDT += dt.getSeconds().toString().padStart(2, '0')
                break;
            case 'w':
                sDT += dt.getDay().toString()
                break;
            default:
                sDT += c;
                break;
        }
    }
    gMe.toolsMgr.workers_postmessage_for_main(gMe.toolsMgr.workers.js, chatid, toolcallid, toolname, sDT);
}


let js_meta = {
        "type": "function",
        "function": {
            "name": "run_javascript_function_code",
            "description": "Runs given code using eval within a web worker context in a browser's javascript environment and returns the console.log outputs of the execution after few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code that will be run using eval within a web worker in the browser's javascript interpreter environment."
                    }
                },
                "required": ["code"]
            }
        }
    }


/**
 * Implementation of the javascript interpretor logic. Minimal skeleton for now.
 * ALERT: Has access to the javascript web worker environment and can mess with it and beyond
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function js_run(chatid, toolcallid, toolname, obj) {
    gMe.toolsMgr.workers.js.postMessage({ cid: chatid, tcid: toolcallid, name: toolname, code: obj["code"]})
}


let calc_meta = {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Calculates the provided arithmatic expression using console.log within a web worker of a browser's javascript interpreter environment and returns the output of the execution once it is done in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "arithexpr":{
                        "type":"string",
                        "description":"The arithmatic expression that will be calculated by passing it to console.log of a browser's javascript interpreter."
                    }
                },
                "required": ["arithexpr"]
            }
        }
    }


/**
 * Implementation of the simple calculator logic. Minimal skeleton for now.
 * ALERT: Has access to the javascript web worker environment and can mess with it and beyond
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function calc_run(chatid, toolcallid, toolname, obj) {
    gMe.toolsMgr.workers.js.postMessage({ cid: chatid, tcid: toolcallid, name: toolname, code: `console.log(${obj["arithexpr"]})`})
}


/**
 * @type {mToolsMgr.TCSwitch}
 */
let tc_switch = {
    "sys_date_time": {
        "handler": sysdatetime_run,
        "meta": sysdatetime_meta,
        "result": ""
    },
    "run_javascript_function_code": {
        "handler": js_run,
        "meta": js_meta,
        "result": ""
    },
    "simple_calculator": {
        "handler": calc_run,
        "meta": calc_meta,
        "result": ""
    },
}


/**
 * Used to get hold of the web worker to use for running tool/function call related code.
 * @param {mChatMagic.Me} me
 */
export async function init(me) {
    gMe = me
}


/**
 * Return the tool call switch with supported / enabled / available tool calls
 * Allows to verify / setup tool calls, which need to cross check things at runtime
 * before getting allowed, like maybe bcas they depend on a config wrt specified
 * chat session.
 * @param {string} chatId
 */
export async function setup(chatId) {
    return tc_switch;
}
