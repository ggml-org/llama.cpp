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
