//@ts-check
// Helpers to work with html elements
// by Humans for All
//


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
 * If innerHTML specified, it takes priority over any innerText specified.
 * @param {string} id
 * @param {(this: HTMLButtonElement, ev: MouseEvent) => any} callback
 * @param {string | undefined} name
 * @param {string | undefined} innerText
 * @param {string | undefined} innerHTML
 */
export function el_create_button(id, callback, name=undefined, innerText=undefined, innerHTML=undefined) {
    if (!name) {
        name = id;
    }
    if (!innerText) {
        innerText = id;
    }
    let btn = document.createElement("button");
    btn.id = id;
    btn.name = name;
    if (innerHTML) {
        btn.innerHTML = innerHTML;
    } else {
        btn.innerText = innerText;
    }
    btn.addEventListener("click", callback);
    return btn;
}

/**
 * Create a para and set it up. Optionaly append it to a passed parent.
 * @param {string} text - assigned to innerText
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


/** @typedef {{true: string, false: string}} BoolToAnyString */

/** @typedef {HTMLButtonElement & {xbool: boolean, xtexts: BoolToAnyString}} HTMLBoolButtonElement */

/**
 * Create a button which represents bool value using specified text wrt true and false.
 * When ever user clicks the button, it will toggle the value and update the shown text.
 *
 * @param {string} id
 * @param {BoolToAnyString} texts
 * @param {boolean} defaultValue
 * @param {function(boolean):void} cb
 */
export function el_create_boolbutton(id, texts, defaultValue, cb) {
    let el = /** @type {HTMLBoolButtonElement} */(document.createElement("button"));
    el.xbool = defaultValue
    el.xtexts = structuredClone(texts)
    el.innerText = el.xtexts[`${defaultValue}`];
    if (id) {
        el.id = id;
    }
    el.addEventListener('click', (ev)=>{
        el.xbool = !el.xbool
        el.innerText = el.xtexts[`${el.xbool}`];
        cb(el.xbool);
    })
    return el;
}


/** @typedef {Object<string, *>} XSelectOptions */

/** @typedef {HTMLSelectElement & {xselected: *, xoptions: XSelectOptions}} HTMLXSelectElement */

/**
 * Create a select ui element, with a set of options to select from.
 * * options: an object which contains name-value pairs
 * * defaultOption: the value whose name should be choosen, by default.
 * * cb : the call back returns the name string of the option selected.
 *
 * @param {string} id
 * @param {XSelectOptions} options
 * @param {*} defaultOption
 * @param {function(string):void} cb
 */
export function el_create_select(id, options, defaultOption, cb) {
    let el = /** @type{HTMLXSelectElement} */(document.createElement("select"));
    el.xselected = defaultOption
    el.xoptions = structuredClone(options)
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
 * Create a div wrapped labeled instance of the passed el.
 *
 * @template {HTMLElement | HTMLInputElement} T
 * @param {string} label
 * @param {T} el
 * @param {string} className
 */
export function el_create_divlabelel(label, el, className="gridx2") {
    let div = document.createElement("div");
    div.className = className;
    let lbl = document.createElement("label");
    lbl.setAttribute("for", el.id);
    lbl.innerText = label;
    div.appendChild(lbl);
    div.appendChild(el);
    return { div: div, el: el };
}


/**
 * Create a div wrapped input of type file,
 * which hides input and shows a button which chains to underlying file type input.
 * @param {string} id
 * @param {string} label
 * @param {string} labelBtnHtml
 * @param {any} defaultValue
 * @param {string} acceptable
 * @param {(arg0: any) => void} cb
 * @param {string} className
 */
export function el_creatediv_inputfilebtn(id, label, labelBtnHtml, defaultValue, acceptable, cb, className) {
    let elX = el_create_divlabelel(label, el_create_input(id, "file", defaultValue, cb), className)
    elX.el.hidden = true;
    elX.el.accept = acceptable
    let idB = `${id}-button`
    let elB = el_create_button(idB, (mev) => {
        elX.el.value = ""
        elX.el.click()
    }, idB, undefined, labelBtnHtml)
    return { div: elX.div, el: elX.el, elB: elB };
}


/**
 * Create a div wrapped input of type file,
 * which hides input and shows a image button which chains to underlying file type input.
 * @param {string} id
 * @param {string} label
 * @param {any} defaultValue
 * @param {string} acceptable
 * @param {(arg0: any) => void} cb
 * @param {string} className
 */
export function el_creatediv_inputfileimgbtn(id, label, defaultValue, acceptable, cb, className) {
    let elX = el_creatediv_inputfilebtn(id, label, `<p>${label}</p>`, defaultValue, acceptable, cb, className);
    let elImg = document.createElement('img')
    elImg.classList.add(`${className}-img`)
    elX.elB.appendChild(elImg)
    return { div: elX.div, el: elX.el, elB: elX.elB, elImg: elImg };
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
    if (propsTreeRoot == "") {
        elFS.id = `ObjPropsEdit-${sLegend.replaceAll(' ', '')}`
        elFS.classList.add('ObjPropsEdit')
    }
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
        let id = `Set${propsTreeRootNew.replaceAll(':','-')}`
        if (((type == "string") || (type == "number"))) {
            let inp = el_create_divlabelel(k, el_create_input(`${id}`, typeDict[type], oObj[k], (val)=>{
                if (type == "number") {
                    val = Number(val);
                }
                oObj[k] = val;
            }));
            if (fRefiner) {
                fRefiner(k, inp.el)
            }
            elFS.appendChild(inp.div);
        } else if (type == "boolean") {
            let bbtn = el_create_divlabelel(k, el_create_boolbutton(`${id}`, {true: "true", false: "false"}, val, (userVal)=>{
                oObj[k] = userVal;
            }));
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
 * Uses recursion to show embedded objects.
 *
 * @param {HTMLDivElement | HTMLElement} elParent
 * @param {any} oObj
 * @param {Array<string>} lProps
 * @param {string} sLegend - the legend/title for the currrent block of properties
 * @param {string} sOffset - can be used to prefix each of the prop entries
 * @param {any | undefined} dClassNames - can specify class for toplegend and remaining levels parent and legend
 */
export function ui_show_obj_props_info(elParent, oObj, lProps, sLegend, sOffset="", dClassNames=undefined) {
    if (sOffset.length == 0) {
        let elDet = document.createElement("details");
        let elSum = document.createElement("summary")
        if (dClassNames && dClassNames['toplegend']) {
            elSum.classList.add(dClassNames['toplegend'])
        }
        elSum.appendChild(document.createTextNode(sLegend))
        sLegend = ""
        elDet.appendChild(elSum)
        elDet.classList.add(`DivObjPropsInfoL${sOffset.length}`)
        elParent.appendChild(elDet)
        elParent = elDet
    }
    let elPLegend = el_create_append_p(sLegend, elParent)
    if ((dClassNames) && (sOffset.length > 0)) {
        if (dClassNames['parent']) {
            elParent.classList.add(dClassNames['parent'])
        }
        if (dClassNames['legend']) {
            elPLegend.classList.add(dClassNames['legend'])
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
            ui_show_obj_props_info(elS, val, Object.keys(val), kPrint, `>${sOffset}`, dClassNames)
            //el_create_append_p(`${k}:${JSON.stringify(oObj[k], null, " - ")}`, elS);
        }
    }
}


/**
 * Remove elements which match specified selectors template
 * @param {string} sSelectorsTemplate
 */
export function remove_els(sSelectorsTemplate) {
    while (true) {
        let el = document.querySelector (sSelectorsTemplate)
        if (!el) {
            return
        }
        el?.remove()
    }
}


/**
 * Get value of specified property belonging to specified css rule and stylesheet.
 * @param {number} ssIndex
 * @param {string} selectorText
 * @param {string} property
 */
export function ss_get(ssIndex, selectorText, property) {
    for (const rule of document.styleSheets[ssIndex].cssRules) {
        if (rule.constructor.name == "CSSStyleRule") {
            let sr = /** @type {CSSStyleRule} */(rule)
            if (sr.selectorText.trim() != selectorText) {
                continue
            }
            // @ts-ignore
            return sr.style[property]
        }
    }
    return undefined
}
