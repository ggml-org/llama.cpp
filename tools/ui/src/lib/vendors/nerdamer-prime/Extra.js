/*
 * Author : Martin Donk
 * Website : http://www.nerdamer.com
 * Email : martin.r.donk@gmail.com
 * License : MIT
 * Source : https://github.com/jiggzson/nerdamer
 */

// Type imports for JSDoc ======================================================
// These typedefs provide type aliases for the interfaces defined in index.d.ts.
// They enable proper type checking when working with the classes defined in this file.
//
// Usage patterns:
// - For return types: @returns {NerdamerSymbolType}
// - For parameters: @param {NerdamerSymbolType} symbol
// - For variable declarations: /** @type {NerdamerSymbolType} */
//
// Note: When casting local class instances to interface types, use the pattern:
//   /** @type {InterfaceType} */ (/** @type {unknown} */ (localInstance))
// This is needed because TypeScript sees local classes and interfaces as separate types.

/**
 * Core type aliases from index.d.ts
 *
 * @typedef {import('./index').NerdamerCore.NerdamerSymbol} NerdamerSymbolType
 *
 * @typedef {import('./index').NerdamerCore.Frac} FracType
 *
 * @typedef {import('./index').NerdamerCore.Vector} VectorType
 *
 * @typedef {import('./index').NerdamerCore.Matrix} MatrixType
 *
 * @typedef {import('./index').NerdamerCore.Parser} ParserType
 *
 * @typedef {import('./index').NerdamerCore.Settings} SettingsType
 *
 * @typedef {import('./index').NerdamerExpression} ExpressionType
 *
 * @typedef {typeof import('./index')} NerdamerType
 *
 * @typedef {import('./index').NerdamerCore.Utils} UtilsInterface
 *
 * @typedef {import('./index').NerdamerCore.Math2} Math2Interface
 *
 * @typedef {import('./index').NerdamerCore.Core} CoreType
 *
 * @typedef {import('./index').ExpressionParam} ExpressionParam
 *
 * @typedef {import('./index').ArithmeticOperand} ArithmeticOperand
 *
 * @typedef {import('./index').NerdamerCore.AlgebraModule} AlgebraModuleType
 *
 * @typedef {import('./index').NerdamerCore.PartFracSubModule} PartFracSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.CalculusModule} CalculusModuleType
 *
 * @typedef {import('./index').NerdamerCore.ExtraModule} ExtraModuleType
 *
 * @typedef {import('./index').NerdamerCore.LaPlaceSubModule} LaPlaceSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.StatisticsSubModule} StatisticsSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.UnitsSubModule} UnitsSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.DecomposeResultObject} DecomposeResultType
 */

// Check if nerdamer exists globally (browser) or needs to be required (Node.js)
let nerdamer = typeof globalThis !== 'undefined' && globalThis.nerdamer ? globalThis.nerdamer : undefined;
if (typeof module !== 'undefined' && nerdamer === undefined) {
    nerdamer = require('./nerdamer.core.js');
    require('./Calculus');
    require('./Algebra');
}

/** @returns {ExtraModuleType} */
(function initExtraModule() {
    /** @type {CoreType} */
    const core = nerdamer.getCore();
    /** @type {ParserType} */
    const _ = core.PARSER;
    const {
        NerdamerSymbol,
        Vector: _Vector,
        /** @type {AlgebraModuleType} */
        Algebra,
        /** @type {CalculusModuleType} */
        Calculus,
    } = core;
    const { format, isVector, isArray, isSymbol } = core.Utils;
    const { S, EX: _EX, CP, PL, CB, FN } = core.groups;
    core.Settings.Laplace_integration_depth = 40;

    /**
     * Check if a symbol's power is itself a symbol with group S or CB
     *
     * @param {NerdamerSymbolType} sym
     * @returns {boolean}
     */
    function hasPowerGroupSOrCB(sym) {
        return isSymbol(sym.power) && (sym.power.group === S || sym.power.group === CB);
    }

    /**
     * Finds a function by name within this symbol's tree.
     *
     * @this {NerdamerSymbolType}
     * @param {string} fname The function name to search for
     * @returns {NerdamerSymbolType | undefined} The found function symbol clone, or undefined if not found
     */
    NerdamerSymbol.prototype.findFunction = function findFunction(fname) {
        // This is what we're looking for
        if (this.group === FN && this.fname === fname) {
            return this.clone();
        }
        let found;
        if (this.symbols) {
            for (const x in this.symbols) {
                if (!Object.hasOwn(this.symbols, x)) {
                    continue;
                }
                found = this.symbols[x].findFunction(fname);
                if (found) {
                    break;
                }
            }
        }

        return found;
    };

    /** @type {ExtraModuleType} */
    const __ = (core.Extra = {
        version: '1.4.2',
        // http://integral-table.com/downloads/LaplaceTable.pdf
        // Laplace assumes all coefficients to be positive
        LaPlace: {
            // Using: integral_0^oo f(t)*e^(-s*t) dt
            /**
             * @param {NerdamerSymbolType} symbol
             * @param {NerdamerSymbolType | string} t
             * @param {NerdamerSymbolType | string} s
             * @returns {NerdamerSymbolType}
             */
            transform(symbol, t, s) {
                /** @type {NerdamerSymbolType} */
                symbol = symbol.clone();

                t = t.toString();
                // First try a lookup for a speed boost
                symbol = NerdamerSymbol.unwrapSQRT(symbol, true);
                /** @type {NerdamerSymbolType} */
                let retval;
                const coeff = symbol.stripVar(t);
                const g = symbol.group;

                symbol = /** @type {NerdamerSymbolType} */ (_.divide(symbol, coeff.clone()));

                if (symbol.isConstant() || !symbol.contains(t, true)) {
                    retval = _.parse(format('({0})/({1})', symbol, s));
                } else if (g === S && core.Utils.isInt(symbol.power)) {
                    const n = String(symbol.power);
                    retval = _.parse(format('factorial({0})/({1})^({0}+1)', n, s));
                } else if (symbol.group === S && symbol.power.equals(1 / 2)) {
                    retval = _.parse(format('sqrt(pi)/(2*({0})^(3/2))', s));
                } else if (symbol.isComposite()) {
                    retval = new NerdamerSymbol(0);
                    symbol.each(x => {
                        retval = /** @type {NerdamerSymbolType} */ (_.add(retval, __.LaPlace.transform(x, t, s)));
                    }, true);
                } else if (symbol.isE() && hasPowerGroupSOrCB(symbol)) {
                    const a = /** @type {NerdamerSymbolType} */ (symbol.power).stripVar(t);
                    retval = _.parse(format('1/(({1})-({0}))', a, s));
                } else {
                    const fns = ['sin', 'cos', 'sinh', 'cosh'];
                    // Support for symbols in fns with arguments in the form a*t or n*t where a = symbolic and n = Number
                    if (
                        symbol.group === FN &&
                        fns.indexOf(symbol.fname) !== -1 &&
                        (symbol.args[0].group === S || symbol.args[0].group === CB)
                    ) {
                        const a = symbol.args[0].stripVar(t);

                        switch (symbol.fname) {
                            case 'sin':
                                retval = _.parse(format('({0})/(({1})^2+({0})^2)', a, s));
                                break;
                            case 'cos':
                                retval = _.parse(format('({1})/(({1})^2+({0})^2)', a, s));
                                break;
                            case 'sinh':
                                retval = _.parse(format('({0})/(({1})^2-({0})^2)', a, s));
                                break;
                            case 'cosh':
                                retval = _.parse(format('({1})/(({1})^2-({0})^2)', a, s));
                                break;
                        }
                    } else {
                        // Try to integrate for a solution
                        // we need at least the Laplace integration depth
                        const depthIsLower = core.Settings.integration_depth < core.Settings.Laplace_integration_depth;

                        let savedIntegrationDepth;
                        if (depthIsLower) {
                            savedIntegrationDepth = core.Settings.integration_depth; // Save the depth
                            core.Settings.integration_depth = core.Settings.Laplace_integration_depth; // Transforms need a little more room
                        }

                        core.Utils.block(
                            'PARSE2NUMBER',
                            () => {
                                const u = t;
                                const sym = symbol.sub(t, u);
                                const integrationExpr = _.parse(`e^(-${s}*${u})*${sym}`);
                                retval = Calculus.integrate(integrationExpr, u);
                                if (retval.hasIntegral?.()) {
                                    retval = _.symfunction('laplace', [symbol, _.parse(String(t)), _.parse(String(s))]);
                                    return;
                                }
                                //                                _.error('Unable to compute transform');
                                retval = retval.sub(t, 0);
                                retval = /** @type {NerdamerSymbolType} */ (
                                    _.expand(_.multiply(retval, new NerdamerSymbol(-1)))
                                );
                                retval = retval.sub(u, t);
                            },
                            false
                        );

                        retval = /** @type {NerdamerSymbolType} */ (
                            core.Utils.block('PARSE2NUMBER', () => _.parse(retval), true)
                        );

                        if (depthIsLower) // Put the integration depth as it was
                        {
                            core.Settings.integration_depth = savedIntegrationDepth;
                        }
                    }
                }

                return /** @type {NerdamerSymbolType} */ (_.multiply(retval, coeff));
            },
            /**
             * @param {NerdamerSymbolType} symbol
             * @param {NerdamerSymbolType | string} s_
             * @param {NerdamerSymbolType | string} t
             * @returns {NerdamerSymbolType}
             */
            inverse(symbol, s_, t) {
                const inputSymbol = symbol.clone();
                return core.Utils.block(
                    'POSITIVE_MULTIPLIERS',
                    () => {
                        /** @type {NerdamerSymbolType | undefined} */
                        let retval;
                        // Expand and get partial fractions
                        if (symbol.group === CB) {
                            symbol = /** @type {NerdamerSymbolType} */ (
                                /** @type {PartFracSubModuleType} */ (Algebra.PartFrac).partfrac(
                                    /** @type {NerdamerSymbolType} */ (_.expand(symbol)),
                                    s_
                                )
                            );
                        }

                        if (symbol.group === S || symbol.group === CB || symbol.isComposite()) {
                            /** @type {number | FracType} */
                            let p;
                            /** @type {FracType} */
                            let denP;
                            /** @type {NerdamerSymbolType} */
                            let a;
                            /** @type {NerdamerSymbolType | string} */
                            let b;
                            /** @type {NerdamerSymbolType} */
                            let d;
                            /** @type {string} */
                            let exp;
                            /** @type {DecomposeResultType} */
                            let f2;
                            /** @type {string | number} */
                            let fact;
                            // Remove the multiplier
                            const m = symbol.multiplier.clone();
                            symbol.toUnitMultiplier();
                            // Get the numerator and denominator
                            let num = symbol.getNum();
                            const den = symbol.getDenom().toUnitMultiplier();

                            // TODO: Make it so factor doesn't destroy pi
                            // num = core.Algebra.Factor.factor(symbol.getNum());
                            // den = core.Algebra.Factor.factor(symbol.getDenom().invert(null, true));

                            if (den.group === CP || den.group === PL) {
                                denP = /** @type {FracType} */ (den.power.clone());
                                den.toLinear();
                            } else {
                                denP = new core.Frac(1);
                            }

                            // Convert s to a string
                            const s = s_.toString();
                            // Split up the denominator if in the form ax+b
                            /** @type {DecomposeResultType} */
                            const f = core.Utils.decompose_fn(den, s, true);
                            // Move the multiplier to the numerator
                            /** @type {DecomposeResultType} */
                            const _fe = core.Utils.decompose_fn(
                                /** @type {NerdamerSymbolType} */ (_.expand(num.clone())),
                                s,
                                true
                            );
                            num.multiplier = num.multiplier.multiply(m);

                            const finalize = function () {
                                // Put back the numerator
                                retval = /** @type {NerdamerSymbolType} */ (_.multiply(retval, num));
                                retval.multiplier = retval.multiplier.multiply(symbol.multiplier);
                                // Put back a
                                retval = /** @type {NerdamerSymbolType} */ (_.divide(retval, f.a));
                            };

                            // Store the parts in variables for easy recognition
                            // check if in the form t^n where n = integer
                            if (
                                (den.group === S || den.group === CB) &&
                                f.x.value === s &&
                                f.b.equals(0) &&
                                core.Utils.isInt(f.x.power)
                            ) {
                                p = /** @type {number} */ (/** @type {unknown} */ (f.x.power)) - 1;
                                fact = core.Math2.factorial(p);
                                //  N!/s^(n-1)
                                retval = /** @type {NerdamerSymbolType} */ (
                                    _.divide(_.pow(_.parse(String(t)), new NerdamerSymbol(p)), new NerdamerSymbol(fact))
                                );
                                // Wrap it up
                                finalize();
                            } else if (den.group === CP && denP.equals(1)) {
                                if (f.x.group === core.groups.PL && Algebra.degree(den).equals(2)) {
                                    // Possibly in the form 1/(s^2+2*s+1)
                                    // Try factoring to get it in a more familiar form{
                                    // Apply inverse of F(s-a)
                                    /**
                                     * @type {{
                                     *     f: NerdamerSymbolType;
                                     *     a: NerdamerSymbolType;
                                     *     h: NerdamerSymbolType;
                                     *     c?: NerdamerSymbolType;
                                     * }}
                                     */
                                    const completed = Algebra.sqComplete(den, s);
                                    const u = core.Utils.getU(den);
                                    // Get a for the function above
                                    a = core.Utils.decompose_fn(completed.a, s, true).b;
                                    const tf = __.LaPlace.inverse(
                                        _.parse(`1/((${u})^2+(${completed.c}))`),
                                        u,
                                        String(t)
                                    );
                                    retval = /** @type {NerdamerSymbolType} */ (
                                        _.multiply(tf, _.parse(`(${m})*e^(-(${a})*(${t}))`))
                                    );
                                    // A/(b*s-c) -> ae^(-bt)
                                } else if (f.x.isLinear() && !num.contains(s)) {
                                    t = /** @type {NerdamerSymbolType | string} */ (
                                        _.divide(_.parse(String(t)), f.a.clone())
                                    );

                                    // Don't add factorial of one or zero
                                    p = /** @type {number} */ (/** @type {unknown} */ (denP)) - 1;
                                    fact = p === 0 || p === 1 ? '1' : `(${denP}-1)!`;
                                    retval = _.parse(
                                        format(
                                            '(({0})^({3}-1)*e^(-(({2})*({0}))/({1})))/(({4})*({1})^({3}))',
                                            t,
                                            f.a,
                                            f.b,
                                            denP,
                                            fact
                                        )
                                    );
                                    // Wrap it up
                                    finalize();
                                } else if (f.x.group === S && f.x.power.equals(2)) {
                                    if (num.contains(s)) {
                                        // A*s/(b*s^2+c^2)
                                        a = new NerdamerSymbol(1);
                                        if (num.group === CB) {
                                            /** @type {NerdamerSymbolType} */
                                            let newNum = new NerdamerSymbol(1);
                                            num.each(x => {
                                                if (x.contains(s)) {
                                                    newNum = /** @type {NerdamerSymbolType} */ (_.multiply(newNum, x));
                                                } else {
                                                    a = /** @type {NerdamerSymbolType} */ (_.multiply(a, x));
                                                }
                                            });
                                            num = newNum;
                                        }

                                        // We need more information about the denominator to decide
                                        f2 = core.Utils.decompose_fn(num, s, true);
                                        const fn1 = f2.a;
                                        const fn2 = f2.b;
                                        const aHasSin = fn1.containsFunction('sin');
                                        const aHasCos = fn1.containsFunction('cos');
                                        const bHasCos = fn2.containsFunction('cos');
                                        const bHasSin = fn2.containsFunction('sin');
                                        if (
                                            f2.x.value === s &&
                                            f2.x.isLinear() &&
                                            !((aHasSin && bHasCos) || aHasCos || bHasSin)
                                        ) {
                                            retval = _.parse(
                                                format(
                                                    '(({1})*cos((sqrt(({2})*({3}))*({0}))/({2})))/({2})',
                                                    t,
                                                    f2.a,
                                                    f.a,
                                                    f.b
                                                )
                                            );
                                        } else if (aHasSin && bHasCos) {
                                            const sin = /** @type {NerdamerSymbolType} */ (fn1.findFunction?.('sin'));
                                            const cos = /** @type {NerdamerSymbolType} */ (fn2.findFunction?.('cos'));
                                            // Who has the s?
                                            if (sin?.args?.[0].equals(cos?.args?.[0]) && !sin?.args?.[0].contains(s)) {
                                                b = /** @type {NerdamerSymbolType} */ (
                                                    _.divide(fn2, cos.toUnitMultiplier())
                                                ).toString();
                                                const c = sin.args[0].toString();
                                                d = f.b;
                                                const e = _.divide(fn1, sin.toUnitMultiplier());
                                                exp =
                                                    '(({1})*({2})*cos({3})*sin(sqrt({4})*({0})))/sqrt({4})+({1})*sin({3})*({5})*cos(sqrt({4})*({0}))';
                                                retval = _.parse(format(exp, t, a, b, c, d, e));
                                            }
                                        }
                                    } else {
                                        retval = _.parse(
                                            format(
                                                '(({1})*sin((sqrt(({2})*({3}))*({0}))/({2})))/sqrt(({2})*({3}))',
                                                t,
                                                num,
                                                f.a,
                                                f.b
                                            )
                                        );
                                    }
                                }
                            } else if (
                                /** @type {FracType} */ (f.x.power).num &&
                                /** @type {FracType} */ (f.x.power).num.equals(3) &&
                                /** @type {FracType} */ (f.x.power).den.equals(2) &&
                                num.contains('sqrt(pi)') &&
                                !num.contains(s) &&
                                num.isLinear()
                            ) {
                                b = /** @type {NerdamerSymbolType} */ (_.divide(num.clone(), _.parse('sqrt(pi)')));
                                retval = _.parse(format('(2*({2})*sqrt({0}))/({1})', t, f.a, b, num));
                            } else if (denP.equals(2) && f.x.power.equals(2)) {
                                if (num.contains(s)) {
                                    // Decompose the numerator to check value of s
                                    f2 = core.Utils.decompose_fn(
                                        /** @type {NerdamerSymbolType} */ (_.expand(num.clone())),
                                        s,
                                        true
                                    );
                                    if (f2.x.isComposite()) {
                                        /** @type {DecomposeResultType[]} */
                                        const sTerms = [];
                                        // First collect the factors e.g. (a)(bx)(cx^2+d)
                                        /** @type {DecomposeResultType[]} */
                                        const symbols = /** @type {DecomposeResultType[]} */ (
                                            num
                                                .collectSymbols(x => {
                                                    x = NerdamerSymbol.unwrapPARENS(x);
                                                    /** @type {DecomposeResultType} */
                                                    const decomp = core.Utils.decompose_fn(x, s, true);
                                                    decomp.symbol = x;
                                                    return decomp;
                                                })
                                                // Then sort them by power hightest to lowest
                                                .sort((x1, x2) => {
                                                    const p1 =
                                                        /** @type {DecomposeResultType} */ (x1).x.value === s
                                                            ? /** @type {number} */ (
                                                                  /** @type {unknown} */ (
                                                                      /** @type {DecomposeResultType} */ (x1).x.power
                                                                  )
                                                              )
                                                            : 0;
                                                    const p2 =
                                                        /** @type {DecomposeResultType} */ (x2).x.value === s
                                                            ? /** @type {number} */ (
                                                                  /** @type {unknown} */ (
                                                                      /** @type {DecomposeResultType} */ (x2).x.power
                                                                  )
                                                              )
                                                            : 0;
                                                    return p2 - p1;
                                                })
                                        );
                                        a = new NerdamerSymbol(-1);
                                        // Grab only the ones which have s
                                        for (let i = 0; i < symbols.length; i++) {
                                            const fc = symbols[i];
                                            if (fc.x.value === s) {
                                                sTerms.push(fc);
                                            } else {
                                                a = /** @type {NerdamerSymbolType} */ (_.multiply(a, fc.symbol));
                                            }
                                        }
                                        // The following 2 assumptions are made
                                        // 1. since the numerator was factored above then each s_term has a unique power
                                        // 2. because the terms are sorted by descending powers then the first item
                                        //    has the highest power
                                        // We can now check for the next type s(s^2-a^2)/(s^2+a^2)^2
                                        if (
                                            sTerms[0].x.power.equals(2) &&
                                            sTerms[1].x.power.equals(1) &&
                                            sTerms[1].b.equals(0) &&
                                            !sTerms[0].b.equals(0)
                                        ) {
                                            b = sTerms[0].a.negate();
                                            exp =
                                                '-(({1})*({2})*({5})*({0})*sin((sqrt(({4})*({5}))*({0}))/({4})))/' +
                                                '(2*({4})^2*sqrt(({4})*({5})))-(({1})*({3})*({0})*sin((sqrt(({4})*({5}))*({0}))/({4})))' +
                                                '/(2*({4})*sqrt(({4})*({5})))+(({1})*({2})*cos((sqrt(({4})*({5}))*({0}))/({4})))/({4})^2';
                                            retval = _.parse(format(exp, t, a, b, sTerms[0].b, f.a, f.b));
                                        }
                                    } else if (f2.x.isLinear()) {
                                        a = /** @type {NerdamerSymbolType} */ (_.divide(f2.a, new NerdamerSymbol(2)));
                                        exp =
                                            '(({1})*({0})*sin((sqrt(({2})*({3}))*({0}))/({2})))/(({2})*sqrt(({2})*({3})))';
                                        retval = _.parse(format(exp, t, a, f.a, f.b));
                                    } else if (f2.x.power.equals(2)) {
                                        if (f2.b.equals(0)) {
                                            a = /** @type {NerdamerSymbolType} */ (
                                                _.divide(f2.a, new NerdamerSymbol(2))
                                            );
                                            exp =
                                                '(({1})*sin((sqrt(({2})*({3}))*({0}))/({2})))/(({2})*sqrt(({2})*({3})))+(({1})*({0})*cos((sqrt(({2})*({3}))*({0}))/({2})))/({2})^2';
                                            retval = _.parse(format(exp, t, a, f.a, f.b));
                                        } else {
                                            a = /** @type {NerdamerSymbolType} */ (
                                                _.divide(f2.a, new NerdamerSymbol(2))
                                            );
                                            d = f2.b.negate();
                                            exp =
                                                '-((({2})*({4})-2*({1})*({3}))*sin((sqrt(({2})*({3}))*({0}))/({2})))/(2*({2})*({3})*sqrt(({2})*({3})))+' +
                                                '(({4})*({0})*cos((sqrt(({2})*({3}))*({0}))/({2})))/(2*({2})*({3}))+(({1})*({0})*cos((sqrt(({2})*({3}))*({0}))/({2})))/({2})^2';
                                            retval = _.parse(format(exp, t, a, f.a, f.b, d));
                                        }
                                    }
                                } else {
                                    a = /** @type {NerdamerSymbolType} */ (_.divide(num, new NerdamerSymbol(2)));
                                    exp =
                                        '(({1})*sin((sqrt(({2})*({3}))*({0}))/({2})))/(({3})*sqrt(({2})*({3})))-(({1})*({0})*cos((sqrt(({2})*({3}))*({0}))/({2})))/(({2})*({3}))';
                                    retval = _.parse(format(exp, t, a, f.a, f.b));
                                }
                            } else if (symbol.isComposite()) {
                                // 1/(s+1)^2
                                if (denP.equals(2) && f.x.group === S) {
                                    retval = _.parse(`(${m})*(${t})*e^(-(${f.b})*(${t}))`);
                                } else {
                                    retval = new NerdamerSymbol(0);

                                    symbol = /** @type {NerdamerSymbolType} */ (
                                        /** @type {PartFracSubModuleType} */ (Algebra.PartFrac).partfrac(
                                            /** @type {NerdamerSymbolType} */ (_.expand(symbol)),
                                            s_
                                        )
                                    );

                                    symbol.each(x => {
                                        retval = /** @type {NerdamerSymbolType} */ (
                                            _.add(retval, __.LaPlace.inverse(x, s_, t))
                                        );
                                    }, true);
                                }
                            }
                        }

                        retval ||= _.symfunction('ilt', [inputSymbol, _.parse(String(s_)), _.parse(String(t))]);

                        return /** @type {NerdamerSymbolType} */ (retval);
                    },
                    true
                );
            },
        },
        Statistics: {
            /**
             * @param {NerdamerSymbolType[]} arr
             * @returns {Record<string, number>}
             */
            frequencyMap(arr) {
                /** @type {Record<string, number>} */
                const map = {};
                // Get the frequency map
                for (let i = 0, l = arr.length; i < l; i++) {
                    const e = arr[i];
                    const key = e.toString();
                    map[key] ||= 0; // Default it to zero
                    map[key]++; // Increment
                }
                return map;
            },
            /**
             * @param {NerdamerSymbolType[]} arr
             * @returns {NerdamerSymbolType[]}
             */
            sort(arr) {
                return arr.sort((a, b) => {
                    if (!a.isConstant() || !b.isConstant()) {
                        _.error('Unable to sort! All values must be numeric');
                    }
                    return /** @type {number} */ (/** @type {unknown} */ (a.multiplier.subtract(b.multiplier)));
                });
            },
            /**
             * @param {NerdamerSymbolType[]} arr
             * @returns {NerdamerSymbolType}
             */
            count(arr) {
                return new NerdamerSymbol(arr.length);
            },
            /**
             * @param {NerdamerSymbolType[]} arr
             * @param {NerdamerSymbolType} [x_]
             * @returns {NerdamerSymbolType}
             */
            sum(arr, x_) {
                /** @type {NerdamerSymbolType} */
                let sum = new NerdamerSymbol(0);
                for (let i = 0, l = arr.length; i < l; i++) {
                    const xi = arr[i].clone();
                    if (x_) {
                        sum = /** @type {NerdamerSymbolType} */ (
                            _.add(_.pow(_.subtract(xi, x_.clone()), new NerdamerSymbol(2)), sum)
                        );
                    } else {
                        sum = /** @type {NerdamerSymbolType} */ (_.add(xi, sum));
                    }
                }

                return sum;
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            mean(...args) {
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.mean(.../** @type {NerdamerSymbolType[]} */ (args[0].elements));
                }
                return /** @type {NerdamerSymbolType} */ (_.divide(__.Statistics.sum(args), __.Statistics.count(args)));
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            median(...args) {
                /** @type {NerdamerSymbolType} */
                let retval;
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.median(.../** @type {NerdamerSymbolType[]} */ (args[0].elements));
                }
                try {
                    const sorted = __.Statistics.sort(args);
                    const l = args.length;
                    if (core.Utils.even(l)) {
                        const mid = l / 2;
                        retval = __.Statistics.mean(sorted[mid - 1], sorted[mid]);
                    } else {
                        retval = sorted[Math.floor(l / 2)];
                    }
                } catch (e) {
                    if (/** @type {Error} */ (e).message === 'timeout') {
                        throw e;
                    }
                    retval = _.symfunction('median', args);
                }
                return retval;
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            mode(...args) {
                /** @type {NerdamerSymbolType} */
                let retval;
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.mode(.../** @type {NerdamerSymbolType[]} */ (args[0].elements));
                }

                const map = __.Statistics.frequencyMap(args);

                // The mode of 1 item is that item as per issue #310 (verified by Happypig375).
                if (core.Utils.keys(map).length === 1) {
                    retval = args[0];
                } else {
                    // Invert by arraning them according to their frequency
                    /** @type {Record<number, string | string[]>} */
                    const inverse = {};
                    for (const x in map) {
                        if (!Object.hasOwn(map, x)) {
                            continue;
                        }
                        const freq = map[x];
                        // Check if it's in the inverse already
                        if (freq in inverse) {
                            const e = inverse[freq];
                            // If it's already an array then just add it
                            if (isArray(e)) {
                                e.push(x);
                            }
                            // Convert it to and array
                            else {
                                inverse[freq] = [x, /** @type {string} */ (inverse[freq])];
                            }
                        } else {
                            inverse[freq] = x;
                        }
                    }
                    // The keys now represent the maxes. We want the max of those keys
                    const keyNums = core.Utils.keys(inverse).map(k => Number(k));
                    const maxKey = Math.max.apply(null, keyNums);
                    const max = inverse[maxKey];
                    // Check it's an array. If it is then map over the results and convert
                    // them to NerdamerSymbol
                    if (isArray(max)) {
                        retval = _.symfunction(
                            'mode',
                            max.sort().map(v => _.parse(v))
                        );
                    } else {
                        retval = _.parse(/** @type {string} */ (max));
                    }
                }

                return retval;
            },
            /**
             * @param {NerdamerSymbolType} k
             * @param {NerdamerSymbolType[]} args
             * @returns {NerdamerSymbolType}
             */
            gVariance(k, args) {
                const x_ = __.Statistics.mean(...args);
                const sum = __.Statistics.sum(args, x_);
                return /** @type {NerdamerSymbolType} */ (_.multiply(k, sum));
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            variance(...args) {
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.variance(.../** @type {NerdamerSymbolType[]} */ (args[0].elements));
                }
                const k = /** @type {NerdamerSymbolType} */ (
                    _.divide(new NerdamerSymbol(1), __.Statistics.count(args))
                );
                return __.Statistics.gVariance(k, args);
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            sampleVariance(...args) {
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.sampleVariance(.../** @type {NerdamerSymbolType[]} */ (args[0].elements));
                }

                const k = /** @type {NerdamerSymbolType} */ (
                    _.divide(new NerdamerSymbol(1), _.subtract(__.Statistics.count(args), new NerdamerSymbol(1)))
                );
                return __.Statistics.gVariance(k, args);
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            standardDeviation(...args) {
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.standardDeviation(.../** @type {NerdamerSymbolType[]} */ (args[0].elements));
                }
                return /** @type {NerdamerSymbolType} */ (
                    _.pow(__.Statistics.variance(...args), new NerdamerSymbol(1 / 2))
                );
            },
            /**
             * @param {...NerdamerSymbolType} args
             * @returns {NerdamerSymbolType}
             */
            sampleStandardDeviation(...args) {
                // Handle arrays
                if (isVector(args[0])) {
                    return __.Statistics.sampleStandardDeviation(
                        .../** @type {NerdamerSymbolType[]} */ (args[0].elements)
                    );
                }
                return /** @type {NerdamerSymbolType} */ (
                    _.pow(__.Statistics.sampleVariance(...args), new NerdamerSymbol(1 / 2))
                );
            },
            /**
             * @param {NerdamerSymbolType} x
             * @param {NerdamerSymbolType} mean
             * @param {NerdamerSymbolType} stdev
             * @returns {NerdamerSymbolType}
             */
            zScore(x, mean, stdev) {
                return /** @type {NerdamerSymbolType} */ (_.divide(_.subtract(x, mean), stdev));
            },
        },
        Units: {
            table: {
                foot: '12 inch',
                meter: '100 cm',
                decimeter: '10 cm',
            },
        },
    });

    nerdamer.register([
        {
            name: 'laplace',
            visible: true,
            numargs: 3,
            build() {
                return __.LaPlace.transform;
            },
        },
        {
            name: 'ilt',
            visible: true,
            numargs: 3,
            build() {
                return __.LaPlace.inverse;
            },
        },
        // Statistical
        {
            name: 'mean',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.mean;
            },
        },
        {
            name: 'median',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.median;
            },
        },
        {
            name: 'mode',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.mode;
            },
        },
        {
            name: 'smpvar',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.sampleVariance;
            },
        },
        {
            name: 'variance',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.variance;
            },
        },
        {
            name: 'smpstdev',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.sampleStandardDeviation;
            },
        },
        {
            name: 'stdev',
            visible: true,
            numargs: -1,
            build() {
                return __.Statistics.standardDeviation;
            },
        },
        {
            name: 'zscore',
            visible: true,
            numargs: 3,
            build() {
                return __.Statistics.zScore;
            },
        },
    ]);

    // Link registered functions externally
    nerdamer.updateAPI();
})();

// Added for all.min.js
if (typeof module !== 'undefined') {
    module.exports = nerdamer;
}
