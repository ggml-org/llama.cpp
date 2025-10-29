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
            "description": "Retrieve the value associated with a given key, in few seconds using a web worker. If key doesnt exist, then __UNDEFINED__ is returned as the value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key whose value should be returned."
                    }
                },
                "required": ["key"],
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
            "description": "Store a value under a given key, in few seconds using a web worker. If the key already exists, its value will be updated to the new value",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key under which to store the value."
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to store, complex objects could be passed in JSON Stringified format."
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
