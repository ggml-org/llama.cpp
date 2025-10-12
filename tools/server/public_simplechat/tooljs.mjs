//@ts-check
// DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
// Helpers to handle tools/functions calling wrt
// * javascript interpreter
// * simple arithmatic calculator
// by Humans for All
//


let gConsoleStr = ""
/**
 * @type { {(...data: any[]): void} | null}
 */
let gOrigConsoleLog = null


/**
 * @param {any[]} args
 */
function console_trapped(...args) {
    let res = args.map((arg)=>{
        if (typeof arg == 'object') {
            return JSON.stringify(arg);
        } else {
            return String(arg);
        }
    }).join(' ');
    gConsoleStr += res;
}

function console_redir() {
    gOrigConsoleLog = console.log
    console.log = console_trapped
    gConsoleStr = ""
}

function console_revert() {
    if (gOrigConsoleLog !== null) {
        console.log = gOrigConsoleLog
    }
}


let js_meta = {
        "type": "function",
        "function": {
            "name": "run_javascript_function_code",
            "description": "Runs given code as a function in a browser's javascript environment and returns the console.log outputs of the execution after few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code belonging to a function to run in the browser's javascript interpreter."
                    }
                },
                "required": ["code"]
            }
        }
    }


/**
 * Implementation of the javascript interpretor logic. Minimal skeleton for now.
 * ALERT: Has access to the javascript environment and can mess with it and beyond
 * @param {any} obj
 */
function js_run(obj) {
    console_redir()
    let func = new Function(obj["code"])
    func()
    console_revert()
    tc_switch["javascript"]["result"] = gConsoleStr
}


let calc_meta = {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Calculates the provided arithmatic expression using console.log of a browser's javascript interpreter and returns the output of the execution once it is done in few seconds",
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
 * ALERT: Has access to the javascript environment and can mess with it and beyond
 * @param {any} obj
 */
function calc_run(obj) {
    console_redir()
    let func = new Function(`console.log(${obj["arithexpr"]})`)
    func()
    console_revert()
    tc_switch["simple_calculator"]["result"] = gConsoleStr
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
    }
}

