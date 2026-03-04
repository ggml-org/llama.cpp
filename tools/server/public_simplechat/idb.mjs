//@ts-check
// Helpers to handle indexedDB provided by browsers
// by Humans for All
//


/**
 * Allows the db connection to be openned.
 * @param {string} dbName
 * @param {string} storeName
 * @param {string} callerTag
 */
export function db_open(dbName, storeName, callerTag="") {
    let tag = `iDB:${callerTag}`
    return new Promise((resolve, reject) => {
        const dbConn = indexedDB.open(dbName, 1);
        dbConn.onupgradeneeded = (ev) => {
            console.debug(`DBUG:${tag}:Conn:Upgrade needed...`)
            dbConn.result.createObjectStore(storeName);
            dbConn.result.onerror = (ev) => {
                console.info(`ERRR:${tag}:Db:Op failed [${ev}]...`)
            }
        };
        dbConn.onsuccess = (ev) => {
            console.debug(`INFO:${tag}:Conn:Opened...`)
            resolve(dbConn.result);
        }
        dbConn.onerror = (ev) => {
            console.info(`ERRR:${tag}:Conn:Failed [${ev}]...`)
            reject(ev);
        }
    });
}


/**
 * Get hold of a transaction wrt a specified store in the db
 * @param {IDBDatabase} db
 * @param {string} storeName
 * @param {IDBTransactionMode} opMode
 */
export function db_trans_store(db, storeName, opMode) {
    let dbTrans = db.transaction(storeName, opMode);
    let dbOS = dbTrans.objectStore(storeName);
    return dbOS
}


/**
 * Put a given key-value pair into a store in a db.
 * Return success or failure through callback.
 *
 * @param {string} dbName
 * @param {string} storeName
 * @param {IDBValidKey} key
 * @param {any} value
 * @param {string | undefined} callerTag
 * @param {(status: boolean, related: IDBValidKey | DOMException | null) => void} cb
 */
export function db_put(dbName, storeName, key, value, callerTag, cb) {
    let tag = `iDB:Put:${callerTag}`;
    db_open(dbName, storeName, tag).then((/** @type {IDBDatabase} */db)=>{
        let reqPut = db_trans_store(db, storeName, 'readwrite').put(value, key)
        reqPut.onerror = (evPut) => {
            console.info(`ERRR:${tag}:OnError:transact failed:${reqPut.error}`)
            cb(false, reqPut.error)
        }
        reqPut.onsuccess = (evPut) => {
            console.info(`DBUG:${tag}:transact success`)
            cb(true, reqPut.result)
        }
    }).catch((errReason)=>{
        console.info(`ERRR:${tag}:Caught:transact failed:${errReason}`)
        cb(false, errReason)
    })
}


/**
 * Return value of specified key from a store in a db,
 * through the provided callback.
 *
 * @param {string} dbName
 * @param {string} storeName
 * @param {IDBValidKey} key
 * @param {string | undefined} callerTag
 * @param {(status: boolean, related: IDBValidKey | DOMException | null) => void} cb
 */
export function db_get(dbName, storeName, key, callerTag, cb) {
    let tag = `iDB:Get:${callerTag}`;
    db_open(dbName, storeName, tag).then((/** @type {IDBDatabase} */db)=>{
        let reqGet = db_trans_store(db, storeName, 'readonly').get(key);
        reqGet.onsuccess = (evGet) => {
            console.info(`DBUG:${tag}:transact success`)
            cb(true, reqGet.result)
        }
        reqGet.onerror = (evGet) => {
            console.info(`ERRR:${tag}:OnError:transact failed:${reqGet.error}`)
            cb(false, reqGet.error)
        }
    }).catch((errReason)=>{
        console.info(`ERRR:${tag}:Caught:transact failed:${errReason}`)
        cb(false, errReason)
    })
}


/**
 * Return all keys from a store in a db,
 * through the provided callback.
 *
 * @param {string} dbName
 * @param {string} storeName
 * @param {string | undefined} callerTag
 * @param {(status: boolean, related: IDBValidKey[] | DOMException | null) => void} cb
 */
export function db_getkeys(dbName, storeName, callerTag, cb) {
    let tag = `iDB:GetKeys:${callerTag}`;
    db_open(dbName, storeName, tag).then((/** @type {IDBDatabase} */db)=>{
        let reqGet = db_trans_store(db, storeName, 'readonly').getAllKeys();
        reqGet.onsuccess = (evGet) => {
            console.info(`DBUG:${tag}:transact success`)
            cb(true, reqGet.result)
        }
        reqGet.onerror = (evGet) => {
            console.info(`ERRR:${tag}:OnError:transact failed:${reqGet.error}`)
            cb(false, reqGet.error)
        }
    }).catch((errReason)=>{
        console.info(`ERRR:${tag}:Caught:transact failed:${errReason}`)
        cb(false, errReason)
    })
}
