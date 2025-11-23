//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling in a direct and dangerous way
// by Humans for All
//


import * as tjs from './tooljs.mjs'
import * as tweb from './toolweb.mjs'
import * as tdb from './tooldb.mjs'
import * as tai from './toolai.mjs'
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

        /**
         * Maintain the latest pending tool call id for each unique chat session id
         * @type {Object<string,string>}
         */
        this.pending = {}

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
        me.defaultCfg.tools.toolNames = []
        await tjs.init(me).then(()=>{
            for (const key in tjs.tc_switch) {
                this.tc_switch[key] = tjs.tc_switch[key]
                me.defaultCfg.tools.toolNames.push(key)
            }
        })
        await tdb.init(me).then(()=>{
            for (const key in tdb.tc_switch) {
                this.tc_switch[key] = tdb.tc_switch[key]
                me.defaultCfg.tools.toolNames.push(key)
            }
        })
        await tai.init(me).then(()=>{
            for (const key in tai.tc_switch) {
                this.tc_switch[key] = tai.tc_switch[key]
                me.defaultCfg.tools.toolNames.push(key)
            }
        })
        let tNs = await tweb.init(me)
        for (const key in tNs) {
            this.tc_switch[key] = tNs[key]
            me.defaultCfg.tools.toolNames.push(key)
        }
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
     * Add specified toolcallid to pending list for specified chat session id.
     * @param {string} chatid
     * @param {string} toolcallid
     */
    toolcallpending_add(chatid, toolcallid) {
        console.debug(`DBUG:ToolsManager:ToolCallPendingAdd:${chatid}:${toolcallid}`)
        this.pending[chatid] = toolcallid;
    }

    /**
     * Clear pending list for specified chat session id.
     * @param {string} chatid
     * @param {string} tag
     */
    toolcallpending_clear(chatid, tag) {
        let curtcid = this.pending[chatid];
        console.debug(`DBUG:ToolsManager:ToolCallPendingClear:${tag}:${chatid}:${curtcid}`)
        delete(this.pending[chatid]);
    }

    /**
     * Check if there is a pending tool call awaiting tool call result for given chat session id.
     * Clears from pending list, if found.
     * @param {string} chatid
     * @param {string} toolcallid
     * @param {string} tag
     */
    toolcallpending_found_cleared(chatid, toolcallid, tag) {
        if (this.pending[chatid] !== toolcallid) {
            console.log(`WARN:ToolsManager:ToolCallPendingFoundCleared:${tag}:${chatid}:${toolcallid} not found, skipping...`)
            return false
        }
        this.toolcallpending_clear(chatid, tag)
        return true
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
                    this.toolcallpending_add(chatid, toolcallid);
                    this.tc_switch[fn]["handler"](chatid, toolcallid, fn, JSON.parse(toolargs))
                    return undefined
                } catch (/** @type {any} */error) {
                    this.toolcallpending_found_cleared(chatid, toolcallid, 'ToolsManager:ToolCall:Exc')
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
        this.workers.js.onmessage = (ev) => {
            if (!this.toolcallpending_found_cleared(ev.data.cid, ev.data.tcid, 'js')) {
                return
            }
            cb(ev.data.cid, ev.data.tcid, ev.data.name, ev.data.data)
        }
        this.workers.db.onmessage = (ev) => {
            if (!this.toolcallpending_found_cleared(ev.data.cid, ev.data.tcid, 'db')) {
                return
            }
            cb(ev.data.cid, ev.data.tcid, ev.data.name, JSON.stringify(ev.data.data, (k,v)=>{
                return (v === undefined) ? '__UNDEFINED__' : v;
            }));
        }
    }

    /**
     * Send message to specified Tools-WebWorker's monitor/onmessage handler of main thread
     * by calling it  directly.
     *
     * The specified web worker's main thread monitor/callback logic is triggerd in a delayed
     * manner by cycling the call through the events loop by using a setTimeout 0, so that the
     * callback gets executed only after the caller's code following the call to this helper
     * is done.
     *
     * NOTE: This is needed to ensure that any tool call handler that returns the tool call
     * result immidiately without using any asynhronous mechanism, doesnt get-messed-by /
     * mess-with the delayed response identifier and rescuer timeout logic.
     *
     * @param {Worker} worker
     * @param {string} chatid
     * @param {string} toolcallid
     * @param {string} toolname
     * @param {string} data
     */
    workers_postmessage_for_main(worker, chatid, toolcallid, toolname, data) {
        let mev = new MessageEvent('message', {data: {cid: chatid, tcid: toolcallid, name: toolname, data: data}});
        setTimeout(function() {
            if (worker.onmessage != null) {
                worker.onmessage(mev)
            }
        }, 0);
    }

}
