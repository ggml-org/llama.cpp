//@ts-check
// STILL DANGER DANGER DANGER - Simple and Stupid - Using from a discardable VM better.
// Helpers to handle db related tool/function calling using web worker
// by Humans for All
//

/**
 * Expects to get a message with cid, tcid, (f)name and args
 * Posts message with cid, tcid, (f)name and data if any
 */


/**
 * Allows the db connection to be openned.
 */
function db_open() {
    return new Promise((resolve, reject) => {
        const dbConn = indexedDB.open('TCDB', 1);
        dbConn.onupgradeneeded = (ev) => {
            console.debug("DBUG:WWDb:Conn:Upgrade needed...")
            dbConn.result.createObjectStore('theDB');
            dbConn.result.onerror = (ev) => {
                console.info(`ERRR:WWDb:Db:Op failed [${ev}]...`)
            }
        };
        dbConn.onsuccess = (ev) => {
            console.debug("INFO:WWDb:Conn:Opened...")
            resolve(dbConn.result);
        }
        dbConn.onerror = (ev) => {
            console.info(`ERRR:WWDb:Conn:Failed [${ev}]...`)
            reject(ev);
        }
    });
}


self.onmessage = async function (ev) {
    try {
        console.info(`DBUG:WWDb:${ev.data.name}:OnMessage started...`)
        /** @type {IDBDatabase} */
        let db = await db_open();
        let dbTrans = db.transaction('theDB', 'readwrite');
        let dbOS = dbTrans.objectStore('theDB');
        let args = ev.data.args;
        switch (ev.data.name) {
            case 'data_store_get':
                let reqGet = dbOS.get(args['key'])
                reqGet.onsuccess = (evGet) => {
                    console.info(`DBUG:WWDb:${ev.data.name}:transact success`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'ok', 'data': reqGet.result, 'msg': `DataStoreGet:Ok:${args['key']}:${reqGet.result}`}
                    });
                }
                reqGet.onerror = (evGet) => {
                    console.info(`ERRR:WWDb:${ev.data.name}:transact failed:${reqGet.error}`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'error', 'msg': `DataStoreGet:Err:${args['key']}:${reqGet.error}`}
                    });
                }
                break;
            case 'data_store_set':
                let reqSet = dbOS.add(args['value'], args['key']);
                reqSet.onerror = (evSet) => {
                    console.info(`ERRR:WWDb:${ev.data.name}:transact failed:${reqSet.error}`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'error', 'msg': `DataStoreSet:Err:${args['key']}:${reqSet.error}`}
                    });
                }
                reqSet.onsuccess = (evSet) => {
                    console.info(`DBUG:WWDb:${ev.data.name}:transact success`)
                    self.postMessage({
                        cid: ev.data.cid,
                        tcid: ev.data.tcid,
                        name: ev.data.name,
                        data: { 'status': 'ok', 'msg': `DataStoreSet:Ok:${args['key']}:${reqSet.result}`}
                    });
                }
                break;
            default:
                console.info(`ERRR:WWDb:${ev.data.name}:OnMessage:Unknown func call...`)
                break;
        }
        console.info(`DBUG:WWDb:${ev.data.name}:OnMessage end`)
    } catch (/** @type {any} */error) {
        let errMsg = `\n\nTool/Function call "${ev.data.name}" raised an exception:${error.name}:${error.message}\n\n`;
        self.postMessage({
            cid: ev.data.cid,
            tcid: ev.data.tcid,
            name: ev.data.name,
            data: {'status': 'error', 'msg': errMsg}
        });
        console.info(`ERRR:WWDb:${ev.data.name}:OnMessage end:${error}`)
    }
}
