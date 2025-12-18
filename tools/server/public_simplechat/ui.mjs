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
    el["xbool"] = defaultValue;
    el["xtexts"] = structuredClone(texts);
    el.innerText = el["xtexts"][String(defaultValue)];
    if (id) {
        el.id = id;
    }
    el.addEventListener('click', (ev)=>{
        el["xbool"] = !el["xbool"];
        el.innerText = el["xtexts"][String(el["xbool"])];
        cb(el["xbool"]);
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
    el["xselected"] = defaultOption;
    el["xoptions"] = structuredClone(options);
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
