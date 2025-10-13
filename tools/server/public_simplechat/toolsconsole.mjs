//@ts-check
// Helpers to handle tools/functions calling wrt console
// by Humans for All
//


export let gConsoleStr = ""
/**
 * @type { {(...data: any[]): void} | null}
 */
export let gOrigConsoleLog = null


/**
 * @param {any[]} args
 */
export function console_trapped(...args) {
    let res = args.map((arg)=>{
        if (typeof arg == 'object') {
            return JSON.stringify(arg);
        } else {
            return String(arg);
        }
    }).join(' ');
    gConsoleStr += res;
}

export function console_redir() {
    gOrigConsoleLog = console.log
    console.log = console_trapped
    gConsoleStr = ""
}

export function console_revert() {
    if (gOrigConsoleLog !== null) {
        console.log = gOrigConsoleLog
    }
}
