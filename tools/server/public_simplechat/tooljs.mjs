//@ts-check
// Helpers to handle tools/functions calling
// by Humans for All
//


let metas = [
    {
        "type":"function",
        "function":{
            "name": "javascript",
            "description":"Runs code in an javascript interpreter and returns the result of the execution after 60 seconds.",
            "parameters":{
                "type":"object",
                "properties":{
                    "code":{
                        "type":"string",
                        "description":"The code to run in the javascript interpreter."
                    }
                },
                "required":["code"]
            }
        }
    }
]


/**
 * Implementation of the javascript interpretor logic. Minimal skeleton for now.
 * @param {any} obj
 */
function tool_run(obj) {
    let func = new Function(obj["code"])
    func()
}

let tswitch = {
    "javascript": tool_run,
}

