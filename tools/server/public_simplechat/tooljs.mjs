//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling wrt
// * javascript interpreter
// * simple arithmatic calculator
// by Humans for All
//


let gToolsWorker = /** @type{Worker} */(/** @type {unknown} */(null));


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
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function js_run(toolcallid, toolname, obj) {
    gToolsWorker.postMessage({ id: toolcallid, name: toolname, code: obj["code"]})
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
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function calc_run(toolcallid, toolname, obj) {
    gToolsWorker.postMessage({ id: toolcallid, name: toolname, code: `console.log(${obj["arithexpr"]})`})
}


/**
 * @type {Object<string, Object<string, any>>}
 */
export let tc_switch = {
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
 * Used to get hold of the web worker to use for running tool/function call related code
 * Also to setup tool calls, which need to cross check things at runtime
 * @param {Worker} toolsWorker
 */
export async function init(toolsWorker) {
    gToolsWorker = toolsWorker
}
