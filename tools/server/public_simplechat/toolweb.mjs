//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling related to web access, pdf, etal
// which work in sync with the bundled simpleproxy.py server logic.
// by Humans for All
//


let gToolsWorker = /** @type{Worker} */(/** @type {unknown} */(null));


/**
 * Send a message to Tools WebWorker's monitor in main thread directly
 * @param {MessageEvent<any>} mev
 */
function message_toolsworker(mev) {
    // @ts-ignore
    gToolsWorker.onmessage(mev)
}


/**
 * Retrieve the global Me instance
 */
function get_gme() {
    return (/** @type {Object<string, Object<string, any>>} */(/** @type {unknown} */(document)))['gMe']
}


/**
 * For now hash the shared secret with the year.
 */
function bearer_transform() {
    let data = `${new Date().getUTCFullYear()}${get_gme().tools.proxyAuthInsecure}`
    return crypto.subtle.digest('sha-256', new TextEncoder().encode(data)).then(ab=>{
        return Array.from(new Uint8Array(ab)).map(b=>b.toString(16).padStart(2,'0')).join('')
    })
}

/**
 * Helper http get logic wrt the bundled SimpleProxy server,
 * which helps execute a given proxy dependent tool call.
 * Expects the simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a predefined query token and value wrt a predefined path
 * NOTE: Initial go, handles textual data type.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 * @param {string} path
 */
async function proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, path) {
    if (gToolsWorker.onmessage != null) {
        let params = new URLSearchParams(obj)
        let newUrl = `${get_gme().tools.proxyUrl}/${path}?${params}`
        let btoken = await bearer_transform()
        fetch(newUrl, { headers: { 'Authorization': `Bearer ${btoken}` }}).then(resp => {
            if (!resp.ok) {
                throw new Error(`${resp.status}:${resp.statusText}`);
            }
            return resp.text()
        }).then(data => {
            message_toolsworker(new MessageEvent('message', {data: {cid: chatid, tcid: toolcallid, name: toolname, data: data}}))
        }).catch((err)=>{
            message_toolsworker(new MessageEvent('message', {data: {cid: chatid, tcid: toolcallid, name: toolname, data: `Error:${err}`}}))
        })
    }
}


/**
 * Setup a proxy server dependent tool call
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {string} tag
 * @param {string} tcPath
 * @param {string} tcName
 * @param {{ [x: string]: any; }} tcsData
 * @param {Object<string, Object<string, any>>} tcs
 */
async function proxyserver_tc_setup(tag, tcPath, tcName, tcsData, tcs) {
    await fetch(`${get_gme().tools.proxyUrl}/aum?url=${tcPath}.jambudweepe.akashaganga.multiverse.987654321123456789`).then(resp=>{
        if (resp.statusText != 'bharatavarshe') {
            console.log(`WARN:ToolWeb:${tag}:Dont forget to run the bundled local.tools/simpleproxy.py to enable me`)
            return
        } else {
            console.log(`INFO:ToolWeb:${tag}:Enabling...`)
        }
        tcs[tcName] = tcsData;
    }).catch(err=>console.log(`WARN:ToolWeb:${tag}:ProxyServer missing?:${err}\nDont forget to run the bundled local.tools/simpleproxy.py`))
}


//
// Fetch Web Url Raw
//


let fetchweburlraw_meta = {
        "type": "function",
        "function": {
            "name": "fetch_web_url_raw",
            "description": "Fetch the requested web url through a proxy server and return the got content as is, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the web page to fetch from the internet"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch web url raw logic.
 * Expects a simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt the path urlraw
 *     which gives the actual url to fetch
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchweburlraw_run(chatid, toolcallid, toolname, obj) {
    // maybe filter out any key other than 'url' in obj
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urlraw');
}


/**
 * Setup fetch_web_url_raw for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function fetchweburlraw_setup(tcs) {
    return proxyserver_tc_setup('FetchWebUrlRaw', 'urlraw', 'fetch_web_url_raw', {
        "handler": fetchweburlraw_run,
        "meta": fetchweburlraw_meta,
        "result": ""
    }, tcs);
}


//
// Fetch Web Url Text
//


let fetchweburltext_meta = {
        "type": "function",
        "function": {
            "name": "fetch_web_url_text",
            "description": "Fetch the requested web url through a proxy server and return its text content after stripping away the html tags as well as head, script, style, header, footer, nav blocks, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"url of the page that will be fetched from the internet and inturn unwanted stuff stripped from its contents to some extent"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the fetch web url text logic.
 * Expects a simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt urltext path,
 *     which gives the actual url to fetch
 * * strips out head as well as any script, style, header, footer, nav and so blocks in body
 *   before returning remaining body contents.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchweburltext_run(chatid, toolcallid, toolname, obj) {
    // maybe filter out any key other than 'url' in obj
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urltext');
}


/**
 * Setup fetch_web_url_text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function fetchweburltext_setup(tcs) {
    return proxyserver_tc_setup('FetchWebUrlText', 'urltext', 'fetch_web_url_text', {
        "handler": fetchweburltext_run,
        "meta": fetchweburltext_meta,
        "result": ""
    }, tcs);
}


//
// Search Web Text
//


let searchwebtext_meta = {
        "type": "function",
        "function": {
            "name": "search_web_text",
            "description": "search web for given words and return the plain text content after stripping the html tags as well as head, script, style, header, footer, nav blocks from got html result page, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "words":{
                        "type":"string",
                        "description":"the words to search for on the web"
                    }
                },
                "required": ["words"]
            }
        }
    }


/**
 * Implementation of the search web text logic. Initial go.
 * Builds on urltext path of the bundled simpleproxy.py.
 * Expects simpleproxy.py server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt urltext path,
 *     which gives the actual url to fetch
 * * strips out head as well as any script, style, header, footer, nav and so blocks in body
 *   before returning remaining body contents.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function searchwebtext_run(chatid, toolcallid, toolname, obj) {
    if (gToolsWorker.onmessage != null) {
        /** @type {string} */
        let searchUrl = get_gme().tools.searchUrl;
        searchUrl = searchUrl.replace("SEARCHWORDS", encodeURIComponent(obj.words));
        delete(obj.words)
        obj['url'] = searchUrl
        return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'urltext');
    }
}


/**
 * Setup search_web_text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function searchwebtext_setup(tcs) {
    return proxyserver_tc_setup('SearchWebText', 'urltext', 'search_web_text', {
        "handler": searchwebtext_run,
        "meta": searchwebtext_meta,
        "result": ""
    }, tcs);
}


//
// Pdf2Text
//


let pdf2text_meta = {
        "type": "function",
        "function": {
            "name": "pdf2text",
            "description": "Read pdf from requested local file path / web url through a proxy server and return its text content after converting pdf to text, in few seconds",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":{
                        "type":"string",
                        "description":"local file path (file://) / web (http/https) based url of the pdf that will be got and inturn converted to text to an extent"
                    }
                },
                "required": ["url"]
            }
        }
    }


/**
 * Implementation of the pdf to text logic.
 * Expects a simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a query token named url wrt pdf2text path,
 *     which gives the actual url to fetch
 * * gets the requested pdf and converts to text, before returning same.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} chatid
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function pdf2text_run(chatid, toolcallid, toolname, obj) {
    return proxyserver_get_anyargs(chatid, toolcallid, toolname, obj, 'pdf2text');
}


/**
 * Setup pdf2text for tool calling
 * NOTE: Currently the logic is setup for the bundled simpleproxy.py
 * @param {Object<string, Object<string, any>>} tcs
 */
async function pdf2text_setup(tcs) {
    return proxyserver_tc_setup('Pdf2Text', 'pdf2text', 'pdf2text', {
        "handler": pdf2text_run,
        "meta": pdf2text_meta,
        "result": ""
    }, tcs);
}



/**
 * Used to get hold of the web worker to use for running tool/function call related code
 * Also to setup tool calls, which need to cross check things at runtime
 * @param {Worker} toolsWorker
 */
export async function init(toolsWorker) {
    /**
     * @type {Object<string, Object<string, any>>} tcs
     */
    let tc_switch = {}
    gToolsWorker = toolsWorker
    await fetchweburlraw_setup(tc_switch)
    await fetchweburltext_setup(tc_switch)
    await searchwebtext_setup(tc_switch)
    await pdf2text_setup(tc_switch)
    return tc_switch
}
