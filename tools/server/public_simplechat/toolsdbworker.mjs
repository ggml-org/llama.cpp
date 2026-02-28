//@ts-check
// STILL DANGER DANGER DANGER - Simple and Stupid - Using from a discardable VM better.
// Helpers to handle db related tool/function calling using web worker
// by Humans for All
//

import * as mIdb from './idb.mjs'


/**
 * Expects to get a message with cid, tcid, (f)name and args
 * Posts message with cid, tcid, (f)name and data if any
 */


self.onmessage = async function (ev) {
    try {
        console.info(`DBUG:WWDb:${ev.data.name}:OnMessage started...`)
        /** @type {IDBDatabase} */
        let db = await mIdb.db_open("TCDB", "theDB", "WWDb");
        let dbOS = mIdb.db_trans_store(db, "theDB", 'readwrite');
        let args = ev.data.args;
        switch (ev.data.name) {

            case 'data_store_list':
                let reqList = dbOS.getAllKeys()
                reqList.onsuccess = (evList) => {
                    console.info(`DBUG:WWDb:${ev.data.name}:transact success`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'ok', 'data': reqList.result, 'msg': `DataStoreList:Ok:NumOfKeys:${reqList.result.length}`}
                    });
                }
                reqList.onerror = (evList) => {
                    console.info(`ERRR:WWDb:${ev.data.name}:transact failed:${reqList.error}`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'error', 'msg': `DataStoreList:Err:${reqList.error}`}
                    });
                }
                break;

            case 'data_store_get':
                let reqGet = dbOS.get(args['key'])
                reqGet.onsuccess = (evGet) => {
                    console.info(`DBUG:WWDb:${ev.data.name}:transact success`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'ok', 'data': reqGet.result, 'msg': `DataStoreGet:Ok:Key:${args['key']}:DataLen:${reqGet.result.length}`}
                    });
                }
                reqGet.onerror = (evGet) => {
                    console.info(`ERRR:WWDb:${ev.data.name}:transact failed:${reqGet.error}`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'error', 'msg': `DataStoreGet:Err:Key:${args['key']}:${reqGet.error}`}
                    });
                }
                break;

            case 'data_store_set':
                let reqSet = dbOS.put(args['value'], args['key']);
                reqSet.onerror = (evSet) => {
                    console.info(`ERRR:WWDb:${ev.data.name}:transact failed:${reqSet.error}`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'error', 'msg': `DataStoreSet:Err:Key:${args['key']}:${reqSet.error}`}
                    });
                }
                reqSet.onsuccess = (evSet) => {
                    console.info(`DBUG:WWDb:${ev.data.name}:transact success`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'ok', 'msg': `DataStoreSet:Ok:Key:${args['key']}:SetKey:${reqSet.result}`}
                    });
                }
                break;

            case 'data_store_delete':
                let reqDel = dbOS.delete(args['key'])
                reqDel.onsuccess = (evDel) => {
                    console.info(`DBUG:WWDb:${ev.data.name}:transact success`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'ok', 'msg': `DataStoreDelete:Ok:Key:${args['key']}:${reqDel.result}`}
                    });
                }
                reqDel.onerror = (evDel) => {
                    console.info(`ERRR:WWDb:${ev.data.name}:transact failed:${reqDel.error}`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'error', 'msg': `DataStoreDelete:Err:Key:${args['key']}:${reqDel.error}`}
                    });
                }
                break;

            default:
                console.info(`ERRR:WWDb:${ev.data.name}:OnMessage:Unknown func call...`)
                break;

        }
        console.info(`DBUG:WWDb:${ev.data.name}:OnMessage end`)
    } catch (/** @type {any} */error) {
        let errMsg = `\nTool/Function call "${ev.data.name}" raised an exception:${error.name}:${error.message}\n`;
        self.postMessage({
            cid: ev.data.cid,
            tcid: ev.data.tcid,
            name: ev.data.name,
            data: {'status': 'error', 'msg': errMsg}
        });
        console.info(`ERRR:WWDb:${ev.data.name}:OnMessage end:${error}`)
    }
}
