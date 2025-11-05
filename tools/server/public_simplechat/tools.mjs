//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'
import * as tweb from './toolweb.mjs'
import * as tdb from './tooldb.mjs'
import * as mChatMagic from './simplechat.js'



export class ToolsManager {

    constructor() {
        /**
         * Maintain currently available tool/function calls
         * @type {Object<string,Object<string,any>>}
         */
        this.tc_switch = {}

        this.workers = {
            js: /** @type {Worker} */(/** @type {unknown} */(undefined)),
            db: /** @type {Worker} */(/** @type {unknown} */(undefined)),
        }

    }

    setup_workers() {
        this.workers.js =  new Worker('./toolsworker.mjs', { type: 'module' });
        this.workers.db = new Worker('./toolsdbworker.mjs', { type: 'module' });
    }

    /**
     * Initialise the ToolsManager,
     * including all the different tools groups.
     * @param {mChatMagic.Me} me
     */
    async init(me) {
        this.setup_workers();
        /**
         * @type {string[]}
         */
        let toolNames = []
        await tjs.init(me).then(()=>{
            for (const key in tjs.tc_switch) {
                this.tc_switch[key] = tjs.tc_switch[key]
                toolNames.push(key)
            }
        })
        await tdb.init(me).then(()=>{
            for (const key in tdb.tc_switch) {
                this.tc_switch[key] = tdb.tc_switch[key]
                toolNames.push(key)
            }
        })
        let tNs = await tweb.init(me)
        for (const key in tNs) {
            this.tc_switch[key] = tNs[key]
            toolNames.push(key)
        }
        return toolNames
    }

    /**
     * Prepare the tools meta data that can be passed to the ai server.
     */
    meta() {
        let tools = []
        for (const key in this.tc_switch) {
            tools.push(this.tc_switch[key]["meta"])
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
    async tool_call(chatid, toolcallid, toolname, toolargs) {
        for (const fn in this.tc_switch) {
            if (fn == toolname) {
                try {
                    this.tc_switch[fn]["handler"](chatid, toolcallid, fn, JSON.parse(toolargs))
                    return undefined
                } catch (/** @type {any} */error) {
                    return `Tool/Function call raised an exception:${error.name}:${error.message}`
                }
            }
        }
        return `Unknown Tool/Function Call:${toolname}`
    }

    /**
     * Setup the callback that will be called when ever message
     * is recieved from the Tools Web Workers.
     * @param {(chatId: string, toolCallId: string, name: string, data: string) => void} cb
     */
    workers_cb(cb) {
        this.workers.js.onmessage = function (ev) {
            cb(ev.data.cid, ev.data.tcid, ev.data.name, ev.data.data)
        }
        this.workers.db.onmessage = function (ev) {
            cb(ev.data.cid, ev.data.tcid, ev.data.name, JSON.stringify(ev.data.data, (k,v)=>{
                return (v === undefined) ? '__UNDEFINED__' : v;
            }));
        }
    }

    /**
     * Send a message to specified tools web worker's monitor in main thread directly
     * @param {Worker} worker
     * @param {string} chatid
     * @param {string} toolcallid
     * @param {string} toolname
     * @param {string} data
     */
    workers_postmessage_for_main(worker, chatid, toolcallid, toolname, data) {
        let mev = new MessageEvent('message', {data: {cid: chatid, tcid: toolcallid, name: toolname, data: data}});
        if (worker.onmessage != null) {
            worker.onmessage(mev)
        }
    }

}
