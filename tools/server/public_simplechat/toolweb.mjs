//@ts-check
// ALERT - Simple Stupid flow - Using from a discardable VM is better
// Helpers to handle tools/functions calling related to web access
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
 * Helper http get logic wrt the bundled SimpleProxy server,
 * which helps execute a given proxy dependent tool call.
 * Expects the simple minded proxy server to be running locally
 * * listening on a configured port
 * * expecting http requests
 *   * with a predefined query token and value wrt a predefined path
 * NOTE: Initial go, handles textual data type.
 * ALERT: Accesses a seperate/external web proxy/caching server, be aware and careful
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 * @param {string} path
 * @param {string} qkey
 * @param {string} qvalue
 */
function proxyserver_get_1arg(toolcallid, toolname, obj, path, qkey, qvalue) {
    if (gToolsWorker.onmessage != null) {
        let newUrl = `${get_gme().tools.fetchProxyUrl}/${path}?${qkey}=${qvalue}`
        fetch(newUrl).then(resp => {
            if (!resp.ok) {
                throw new Error(`${resp.status}:${resp.statusText}`);
            }
            return resp.text()
        }).then(data => {
            message_toolsworker(new MessageEvent('message', {data: {id: toolcallid, name: toolname, data: data}}))
        }).catch((err)=>{
            message_toolsworker(new MessageEvent('message', {data: {id: toolcallid, name: toolname, data: `Error:${err}`}}))
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
    await fetch(`${get_gme().tools.fetchProxyUrl}/aum?url=${tcPath}.jambudweepe.akashaganga.multiverse.987654321123456789`).then(resp=>{
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
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchweburlraw_run(toolcallid, toolname, obj) {
    return proxyserver_get_1arg(toolcallid, toolname, obj, 'urlraw', 'url', encodeURIComponent(obj.url));
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
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function fetchweburltext_run(toolcallid, toolname, obj) {
    return proxyserver_get_1arg(toolcallid, toolname, obj, 'urltext', 'url', encodeURIComponent(obj.url));
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
 * @param {string} toolcallid
 * @param {string} toolname
 * @param {any} obj
 */
function searchwebtext_run(toolcallid, toolname, obj) {
    if (gToolsWorker.onmessage != null) {
        /** @type {string} */
        let searchUrl = get_gme().tools.searchUrl;
        searchUrl = searchUrl.replace("SEARCHWORDS", encodeURIComponent(obj.words));
        return proxyserver_get_1arg(toolcallid, toolname, obj, 'urltext', 'url', encodeURIComponent(searchUrl));
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
    return tc_switch
}
