//@ts-check
// DANGER DANGER DANGER - Simple and Stupid - Use from a discardable VM only
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
 * Send a message to Tools WebWorker's monitor in main thread directly 
 * @param {MessageEvent<any>} mev
 */
function message_toolsworker(mev) {
    // @ts-ignore
    gToolsWorker.onmessage(mev)
}


let weburlfetch_meta = {
        "type": "function",
        "function": {
            "name": "web_url_fetch",
            "description": "Fetch the requested web url through a proxy server in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"the url of the page / content to fetch from the internet"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the web url logic. Dumb initial go.
 * Expects a simple minded proxy server to be running locally
 * * listening on port 3128
 * * expecting http requests
 *   * with a query token named url which gives the actual url to fetch
 * ALERT: Accesses a external web proxy/caching server be aware and careful
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function weburlfetch_run(toolcallid, toolname, obj) {
    if (gToolsWorker.onmessage != null) {
        let newUrl = `http://127.0.0.1:3128/urlraw?url=${obj.url}`
        fetch(newUrl).then(resp=>resp.text()).then(data => {
            message_toolsworker(new MessageEvent('message', {data: {id: toolcallid, name: toolname, data: data}}))
        }).catch((err)=>{
            message_toolsworker(new MessageEvent('message', {data: {id: toolcallid, name: toolname, data: `Error:${err}`}}))
        })
    }
}


let weburlfetchstrip_meta = {
        "type": "function",
        "function": {
            "name": "web_url_fetch_strip",
            "description": "Fetch the requested web url through a proxy server and strip away head, script, styles blocks before sending remaining body in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"the url of the page that will be fetched from the internet and inturn contents stripped to some extent"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the web url logic. Dumb initial go.
 * Expects a simple minded proxy server to be running locally
 * * listening on port 3128
 * * expecting http requests
 *   * with a query token named url which gives the actual url to fetch
 * * strips out head as well as any script and style blocks in body
 *   before returning remaining body contents.
 * ALERT: Accesses a external web proxy/caching server be aware and careful
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function weburlfetchstrip_run(toolcallid, toolname, obj) {
    if (gToolsWorker.onmessage != null) {
        let newUrl = `http://127.0.0.1:3128/urltext?url=${obj.url}`
        fetch(newUrl).then(resp=>resp.text()).then(data => {
            message_toolsworker(new MessageEvent('message', {data: {id: toolcallid, name: toolname, data: data}}))
        }).catch((err)=>{
            message_toolsworker(new MessageEvent('message', {data: {id: toolcallid, name: toolname, data: `Error:${err}`}}))
        })
    }
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
    "web_url_fetch": {
        "handler": weburlfetch_run,
        "meta": weburlfetch_meta,
        "result": ""
    },
    "web_url_fetch_strip": {
        "handler": weburlfetchstrip_run,
        "meta": weburlfetchstrip_meta,
        "result": ""
    }
}


/**
 * Used to get hold of the web worker to use for running tool/function call related code
 * @param {Worker} toolsWorker
 */
export function init(toolsWorker) {
    gToolsWorker = toolsWorker
}
