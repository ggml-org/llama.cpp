//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling wrt data store
// using a web worker.
// by Humans for All
//


let gToolsDBWorker = /** @type{Worker} */(/** @type {unknown} */(null));


let dsget_meta = {
        "type": "function",
        "function": {
            "name": "data_store_get",
            "description": "Retrieve the value associated with a given key, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key whose value should be returned."
                    }
                },
                "required": [ "key" ],
            }
        }
    }


/**
 * Implementation of the data store get logic. Minimal skeleton for now.
 * NOTE: Has access to the javascript web worker environment and can mess with it and beyond
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function dsget_run(chatid, toolcallid, toolname, obj) {
    gToolsDBWorker.postMessage({ cid: chatid, tcid: toolcallid, name: toolname, args: obj})
}


let dsset_meta = {
        "type": "function",
        "function": {
            "name": "data_store_set",
            "description": "Store a value under a given key, in few seconds using a web worker",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key under which to store the value."
                    },
                    "value": {
                        "type": "any",
                        "description": "The value to store. Can be any JSON-serialisable type."
                    }
                },
                "required": ["key", "value"]
            },
        }
    }


/**
 * Implementation of the data store set logic. Minimal skeleton for now.
 * NOTE: Has access to the javascript web worker environment and can mess with it and beyond
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function dsset_run(chatid, toolcallid, toolname, obj) {
    gToolsDBWorker.postMessage({ cid: chatid, tcid: toolcallid, name: toolname, args: obj})
}



/**
 * @type {Object<string, Object<string, any>>}
 */
export let tc_switch = {
    "data_store_get": {
        "handler": dsget_run,
        "meta": dsget_meta,
        "result": ""
    },
    "data_store_set": {
        "handler": dsset_run,
        "meta": dsset_meta,
        "result": ""
    },
}


/**
 * Used to get hold of the web worker to use for running tool/function call related code
 * Also to setup tool calls, which need to cross check things at runtime
 * @param {Worker} toolsWorker
 */
export async function init(toolsWorker) {
    gToolsDBWorker = toolsWorker
}
