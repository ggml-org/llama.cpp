//@ts-check
// Helpers to work with html elements
// by Humans for All
//


/**
 * Insert key-value pairs into passed element object.
 * @param {HTMLElement} el
 * @param {string} key
 * @param {any} value
 */
function el_set(el, key, value) {
    // @ts-ignore
    el[key] = value
}

/**
 * Retrieve the value corresponding to given key from passed element object.
 * @param {HTMLElement} el
 * @param {string} key
 */
function el_get(el, key) {
    // @ts-ignore
    return el[key]
}

/**
 * Set the class of the children, based on whether it is the idSelected or not.
 * @param {HTMLDivElement} elBase
 * @param {string} idSelected
 * @param {string} classSelected
 * @param {string} classUnSelected
 */
export function el_children_config_class(elBase, idSelected, classSelected, classUnSelected="") {
    for(let child of elBase.children) {
        if (child.id == idSelected) {
            child.className = classSelected;
        } else {
            child.className = classUnSelected;
        }
    }
}

/**
 * Create button and set it up.
 * @param {string} id
 * @param {(this: HTMLButtonElement, ev: MouseEvent) => any} callback
 * @param {string | undefined} name
 * @param {string | undefined} innerText
 */
export function el_create_button(id, callback, name=undefined, innerText=undefined) {
    if (!name) {
        name = id;
    }
    if (!innerText) {
        innerText = id;
    }
    let btn = document.createElement("button");
    btn.id = id;
    btn.name = name;
    btn.innerText = innerText;
    btn.addEventListener("click", callback);
    return btn;
}

/**
 * Create a para and set it up. Optionaly append it to a passed parent.
 * @param {string} text
 * @param {HTMLElement | undefined} elParent
 * @param {string | undefined} id
 */
export function el_create_append_p(text, elParent=undefined, id=undefined) {
    let para = document.createElement("p");
    para.innerText = text;
    if (id) {
        para.id = id;
    }
    if (elParent) {
        elParent.appendChild(para);
    }
    return para;
}

/**
 * Create a button which represents bool value using specified text wrt true and false.
 * When ever user clicks the button, it will toggle the value and update the shown text.
 *
 * @param {string} id
 * @param {{true: string, false: string}} texts
 * @param {boolean} defaultValue
 * @param {function(boolean):void} cb
 */
export function el_create_boolbutton(id, texts, defaultValue, cb) {
    let el = document.createElement("button");
    el_set(el, "xbool", defaultValue)
    el_set(el, "xtexts", structuredClone(texts))
    el.innerText = el_get(el, "xtexts")[String(defaultValue)];
    if (id) {
        el.id = id;
    }
    el.addEventListener('click', (ev)=>{
        el_set(el, "xbool", !el_get(el, "xbool"));
        el.innerText = el_get(el, "xtexts")[String(el_get(el, "xbool"))];
        cb(el_get(el, "xbool"));
    })
    return el;
}

/**
 * Create a div wrapped button which represents bool value using specified text wrt true and false.
 * @param {string} id
 * @param {string} label
 * @param {{ true: string; false: string; }} texts
 * @param {boolean} defaultValue
 * @param {(arg0: boolean) => void} cb
 * @param {string} className
 */
export function el_creatediv_boolbutton(id, label, texts, defaultValue, cb, className="gridx2") {
    let div = document.createElement("div");
    div.className = className;
    let lbl = document.createElement("label");
    lbl.setAttribute("for", id);
    lbl.innerText = label;
    div.appendChild(lbl);
    let btn = el_create_boolbutton(id, texts, defaultValue, cb);
    div.appendChild(btn);
    return { div: div, el: btn };
}


/**
 * Create a select ui element, with a set of options to select from.
 * * options: an object which contains name-value pairs
 * * defaultOption: the value whose name should be choosen, by default.
 * * cb : the call back returns the name string of the option selected.
 *
 * @param {string} id
 * @param {Object<string,*>} options
 * @param {*} defaultOption
 * @param {function(string):void} cb
 */
export function el_create_select(id, options, defaultOption, cb) {
    let el = document.createElement("select");
    el_set(el, "xselected", defaultOption);
    el_set(el, "xoptions", structuredClone(options));
    for(let cur of Object.keys(options)) {
        let op = document.createElement("option");
        op.value = cur;
        op.innerText = cur;
        if (options[cur] == defaultOption) {
            op.selected = true;
        }
        el.appendChild(op);
    }
    if (id) {
        el.id = id;
        el.name = id;
    }
    el.addEventListener('change', (ev)=>{
        let target = /** @type{HTMLSelectElement} */(ev.target);
        console.log("DBUG:UI:Select:", id, ":", target.value);
        cb(target.value);
    })
    return el;
}

/**
 * Create a div wrapped select ui element, with a set of options to select from.
 *
 * @param {string} id
 * @param {any} label
 * @param {{ [x: string]: any; }} options
 * @param {any} defaultOption
 * @param {(arg0: string) => void} cb
 * @param {string} className
 */
export function el_creatediv_select(id, label, options, defaultOption, cb, className="gridx2") {
    let div = document.createElement("div");
    div.className = className;
    let lbl = document.createElement("label");
    lbl.setAttribute("for", id);
    lbl.innerText = label;
    div.appendChild(lbl);
    let sel = el_create_select(id, options,defaultOption, cb);
    div.appendChild(sel);
    return { div: div, el: sel };
}


/**
 * Create a input ui element.
 *
 * @param {string} id
 * @param {string} type
 * @param {any} defaultValue
 * @param {function(any):void} cb
 */
export function el_create_input(id, type, defaultValue, cb) {
    let el = document.createElement("input");
    el.type = type;
    el.value = defaultValue;
    if (id) {
        el.id = id;
    }
    el.addEventListener('change', (ev)=>{
        cb(el.value);
    })
    return el;
}

/**
 * Create a div wrapped input.
 *
 * @param {string} id
 * @param {string} label
 * @param {string} type
 * @param {any} defaultValue
 * @param {function(any):void} cb
 * @param {string} className
 */
export function el_creatediv_input(id, label, type, defaultValue, cb, className="gridx2") {
    let div = document.createElement("div");
    div.className = className;
    let lbl = document.createElement("label");
    lbl.setAttribute("for", id);
    lbl.innerText = label;
    div.appendChild(lbl);
    let el = el_create_input(id, type, defaultValue, cb);
    div.appendChild(el);
    return { div: div, el: el };
}


/**
 * Auto create ui input elements for specified fields/properties in given object
 * Currently supports text, number, boolean field types.
 * Also supports recursing if a object type field is found.
 *
 * If for any reason the caller wants to refine the created ui element for a specific prop,
 * they can define a fRefiner callback, which will be called back with prop name and ui element.
 * The fRefiner callback even helps work with Obj with-in Obj scenarios.
 *
 * For some reason if caller wants to handle certain properties on their own
 * * specify the prop name of interest along with its prop-tree-hierarchy in lTrapThese
 *   * always start with : when ever refering to propWithPath,
 *     as it indirectly signifies root of properties tree
 *   * remember to seperate the properties tree hierarchy members using :
 * * fTrapper will be called with the parent ui element
 *   into which the new ui elements created for editting the prop, if any, should be attached,
 *   along with the current prop of interest and its full propWithPath representation.
 * @param {HTMLDivElement|HTMLFieldSetElement} elParent
 * @param {string} propsTreeRoot
 * @param {any} oObj
 * @param {Array<string>} lProps
 * @param {string} sLegend
 * @param {((prop:string, elProp: HTMLElement)=>void)| undefined} fRefiner
 * @param {Array<string> | undefined} lTrapThese
 * @param {((propWithPath: string, prop: string, elParent: HTMLFieldSetElement)=>void) | undefined} fTrapper
 */
export function ui_show_obj_props_edit(elParent, propsTreeRoot, oObj, lProps, sLegend, fRefiner=undefined, lTrapThese=undefined, fTrapper=undefined) {
    let typeDict = {
        "string": "text",
        "number": "number",
    };
    let elFS = document.createElement("fieldset");
    let elLegend = document.createElement("legend");
    elLegend.innerText = sLegend;
    elFS.appendChild(elLegend);
    elParent.appendChild(elFS);
    for(const k of lProps) {
        let propsTreeRootNew = `${propsTreeRoot}:${k}`
        if (lTrapThese) {
            if (lTrapThese.indexOf(propsTreeRootNew) != -1) {
                if (fTrapper) {
                    fTrapper(propsTreeRootNew, k, elFS)
                }
                continue
            }
        }
        let val = oObj[k];
        let type = typeof(val);
        if (((type == "string") || (type == "number"))) {
            let inp = el_creatediv_input(`Set${k}`, k, typeDict[type], oObj[k], (val)=>{
                if (type == "number") {
                    val = Number(val);
                }
                oObj[k] = val;
            });
            if (fRefiner) {
                fRefiner(k, inp.el)
            }
            elFS.appendChild(inp.div);
        } else if (type == "boolean") {
            let bbtn = el_creatediv_boolbutton(`Set{k}`, k, {true: "true", false: "false"}, val, (userVal)=>{
                oObj[k] = userVal;
            });
            if (fRefiner) {
                fRefiner(k, bbtn.el)
            }
            elFS.appendChild(bbtn.div);
        } else if (type == "object") {
            ui_show_obj_props_edit(elFS, propsTreeRootNew, val, Object.keys(val), k, (prop, elProp)=>{
                if (fRefiner) {
                    let theProp = `${k}:${prop}`
                    fRefiner(theProp, elProp)
                }
            }, lTrapThese, fTrapper)
        }
    }
}


/**
 * Show the specified properties and their values wrt the given object,
 * with in the elParent provided.
 * @param {HTMLDivElement | HTMLElement} elParent
 * @param {any} oObj
 * @param {Array<string>} lProps
 * @param {string} sLegend
 * @param {string} sOffset - can be used to prefix each of the prop entries
 * @param {any | undefined} dClassNames - can specify class for top level div and legend
 */
export function ui_show_obj_props_info(elParent, oObj, lProps, sLegend, sOffset="", dClassNames=undefined) {
    if (sOffset.length == 0) {
        let div = document.createElement("div");
        div.classList.add(`DivObjPropsInfoL${sOffset.length}`)
        elParent.appendChild(div)
        elParent = div
    }
    let elPLegend = el_create_append_p(sLegend, elParent)
    if (dClassNames) {
        if (dClassNames['div']) {
            elParent.className = dClassNames['div']
        }
        if (dClassNames['legend']) {
            elPLegend.className = dClassNames['legend']
        }
    }
    let elS = document.createElement("section");
    elS.classList.add(`SectionObjPropsInfoL${sOffset.length}`)
    elParent.appendChild(elPLegend);
    elParent.appendChild(elS);

    for (const k of lProps) {
        let kPrint = `${sOffset}${k}`
        let val = oObj[k];
        let vtype = typeof(val)
        if (vtype != 'object') {
            el_create_append_p(`${kPrint}: ${oObj[k]}`, elS)
        } else {
            ui_show_obj_props_info(elS, val, Object.keys(val), kPrint, `>${sOffset}`)
            //el_create_append_p(`${k}:${JSON.stringify(oObj[k], null, " - ")}`, elS);
        }
    }
}
