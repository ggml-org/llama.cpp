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
 * @typedef {NerdamerSymbolType | VectorType | MatrixType} ParseResultType Union type for parse results
 *
 * @typedef {import('./index').NerdamerCore.Parser} ParserType
 *
 * @typedef {import('./index').NerdamerCore.Collection} CollectionType
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
 * @typedef {import('./index').ExpandOptions} ExpandOptions
 *
 * @typedef {import('./index').NerdamerCore.DecomposeResultObject} DecomposeResultType Constructor types
 *
 * @typedef {import('./index').NerdamerCore.FracConstructor} FracConstructor
 *
 * @typedef {import('./index').NerdamerCore.SymbolConstructor} SymbolConstructor
 *
 * @typedef {import('./index').NerdamerCore.VectorConstructor} VectorConstructor
 *
 * @typedef {import('./index').NerdamerCore.AlgebraModule} AlgebraModuleType
 *
 * @typedef {import('./index').NerdamerCore.Polynomial} Polynomial
 *
 * @typedef {import('./index').NerdamerCore.Factors} Factors
 *
 * @typedef {import('./index').NerdamerCore.FactorsLike} FactorsLike
 *
 * @typedef {import('./index').NerdamerCore.MVTerm} MVTerm
 *
 * @typedef {import('./index').NerdamerCore.FactorSubModule} FactorInterface
 *
 * @typedef {import('./index').NerdamerCore.SimplifySubModule} SimplifyInterface
 *
 * @typedef {import('./index').NerdamerCore.PartFracSubModule} PartFracInterface
 *
 * @typedef {new () => Factors} FactorsConstructor
 */

// Check if nerdamer exists globally (browser) or needs to be required (Node.js)
let nerdamer = typeof globalThis !== 'undefined' && globalThis.nerdamer ? globalThis.nerdamer : undefined;
if (typeof module !== 'undefined' && nerdamer === undefined) {
    nerdamer = require('./nerdamer.core.js');
    require('./Calculus.js');
}

(function initAlgebraModule() {
    /* Shortcuts*/
    /** @type {CoreType} */
    const core = nerdamer.getCore();
    /** @type {ParserType} */
    const _ = core.PARSER;
    const { N, P, S, EX, FN, PL, CP, CB } = core.groups;
    const { keys, even, variables, format, round, isInt } = core.Utils;
    const { Frac, NerdamerSymbol, Vector: _Vector, Expression: _Expression } = core;
    const { CONST_HASH } = core.Settings;
    /** @type {Record<string, Function>} */
    const math = core.Utils.importFunctions();
    const _evaluate = core.Utils.evaluate;
    //* ************** CLASSES ***************//
    /**
     * Converts a symbol into an equivalent polynomial arrays of the form [[coefficient_1, power_1],[coefficient_2,
     * power_2], ... ] Univariate polymials only.
     *
     * @class
     * @this {Polynomial}
     * @param {NerdamerSymbolType | number | string} [symbol]
     * @param {string} [variable] The variable name of the polynomial
     * @param {number} [order]
     */
    function Polynomial(symbol, variable, order) {
        /** @type {FracType[]} */
        this.coeffs = [];
        /** @type {string} */
        this.variable = '';

        if (core.Utils.isSymbol(symbol)) {
            this.parse(/** @type {NerdamerSymbolType} */ (symbol));
            this.variable ||= variable || '';
        } else if (typeof symbol === 'number' && !isNaN(symbol)) {
            order ||= 0;
            if (variable === undefined) {
                throw new core.exceptions.InvalidVariableNameError(
                    'Polynomial expects a variable name when creating using order'
                );
            }
            this.coeffs = [];
            this.coeffs[order] = new Frac(symbol);
            this.fill(symbol);
        } else if (typeof symbol === 'string') {
            this.parse(_.parse(symbol));
        }
    }
    /**
     * Creates a Polynomial given an array of coefficients
     *
     * @param {FracType[]} arr
     * @param {string} variable
     * @returns {Polynomial}
     */
    Polynomial.fromArray = function fromArray(arr, variable) {
        if (typeof variable === 'undefined') {
            throw new core.exceptions.InvalidVariableNameError(
                'A variable name must be specified when creating polynomial from array'
            );
        }
        /** @type {Polynomial} */
        const p = new Polynomial();
        p.coeffs = arr;
        p.variable = variable;
        return p;
    };

    /**
     * @param {number} c1
     * @param {number} c2
     * @param {number} n
     * @param {number} base
     * @param {number} p
     * @param {string} variable
     * @returns {Polynomial | null}
     */
    Polynomial.fit = function fit(c1, c2, n, base, p, variable) {
        // After having looped through and mod 10 the number to get the matching factor
        const terms = new Array(p + 1);
        let t = n - c2;
        terms[0] = c2; // The constants is assumed to be correct
        // constant for x^p is also assumed know so add
        terms[p] = c1;
        t -= c1 * base ** p;
        // Start fitting
        for (let i = p - 1; i > 0; i--) {
            const b = base ** i; // We want as many wholes as possible
            const q = t / b;
            const sign = Math.sign(q);
            const c = sign * Math.floor(Math.abs(q));
            t -= c * b;
            terms[i] = c;
        }
        if (t !== 0) {
            return null;
        }
        for (let i = 0; i < terms.length; i++) {
            terms[i] = new Frac(terms[i]);
        }

        return Polynomial.fromArray(terms, variable);
    };

    Polynomial.prototype = {
        /**
         * Converts NerdamerSymbol to Polynomial
         *
         * @this {Polynomial}
         * @param {NerdamerSymbolType} symbol
         * @param {FracType[]} [c] - A collector array
         * @returns {void}
         */
        parse(symbol, c) {
            this.variable = variables(symbol)[0];
            if (!symbol.isPoly()) {
                throw new core.exceptions.NerdamerTypeError(`Polynomial Expected! Received ${core.Utils.text(symbol)}`);
            }
            c ||= [];
            if (!(/** @type {FracType} */ (symbol.power).absEquals(1))) {
                symbol = /** @type {NerdamerSymbolType} */ (_.expand(symbol));
            }

            if (symbol.group === core.groups.N) {
                c[0] = symbol.multiplier;
            } else if (symbol.group === core.groups.S) {
                c[Number(/** @type {FracType} */ (symbol.power).toDecimal())] = symbol.multiplier;
            } else {
                for (const x in symbol.symbols) {
                    if (!Object.hasOwn(symbol.symbols, x)) {
                        continue;
                    }
                    const sub = symbol.symbols[x];
                    const p = sub.power;
                    if (core.Utils.isSymbol(p)) {
                        throw new core.exceptions.NerdamerTypeError('power cannot be a NerdamerSymbol');
                    }

                    const pNum = sub.group === N ? 0 : Number(/** @type {FracType} */ (p).toDecimal());
                    if (sub.symbols) {
                        this.parse(sub, c);
                    } else {
                        c[pNum] = sub.multiplier;
                    }
                }
            }

            this.coeffs = c;

            this.fill();
        },
        /**
         * Fills in the holes in a polynomial with zeroes
         *
         * @this {Polynomial}
         * @param {number} [x] - The number to fill the holes with
         * @returns {Polynomial}
         */
        fill(x) {
            x = Number(x) || 0;
            const l = this.coeffs.length;
            for (let i = 0; i < l; i++) {
                if (this.coeffs[i] === undefined) {
                    this.coeffs[i] = new Frac(x);
                }
            }
            return this;
        },
        /**
         * Removes higher order zeros or a specific coefficient
         *
         * @this {Polynomial}
         * @returns {Polynomial}
         */
        trim() {
            let l = this.coeffs.length;
            while (l--) {
                const c = this.coeffs[l];
                const equalsZero = c.equals(0);
                if (c && equalsZero) {
                    if (l === 0) {
                        break;
                    }
                    this.coeffs.pop();
                } else {
                    break;
                }
            }

            return this;
        },
        /**
         * Returns polynomial mod p **currently fails**
         *
         * @this {Polynomial}
         * @param {number} p
         * @returns {Polynomial}
         */
        modP(p) {
            const l = this.coeffs.length;
            for (let i = 0; i < l; i++) {
                let c = this.coeffs[i];
                let j;
                if (c.lessThan(0)) {
                    // Go borrow
                    /** @type {FracType | undefined} */
                    let b; // A coefficient > 0
                    for (j = i; j < l; j++) {
                        // Starting from where we left off
                        if (this.coeffs[j].greaterThan(0)) {
                            b = this.coeffs[j];
                            break;
                        }
                    }

                    if (b) {
                        // If such a coefficient exists
                        for (; j > i; j--) {
                            // Go down the line and adjust using p
                            this.coeffs[j] = this.coeffs[j].subtract(new Frac(1));
                            this.coeffs[j - 1] = this.coeffs[j - 1].add(new Frac(p));
                        }
                        c = this.coeffs[i]; // Reset c
                    }
                }

                const d = c.mod(new Frac(p));
                const w = c.subtract(d).divide(new Frac(p));
                if (!w.equals(0)) {
                    const upOne = i + 1;
                    let next = this.coeffs[upOne] || new Frac(0);
                    next = next.add(w);
                    this.coeffs[upOne] = next;
                    this.coeffs[i] = d;
                }
            }

            return this;
        },
        /**
         * Adds together 2 polynomials
         *
         * @this {Polynomial}
         * @param {Polynomial} poly
         * @returns {Polynomial}
         */
        add(poly) {
            const l = Math.max(this.coeffs.length, poly.coeffs.length);
            for (let i = 0; i < l; i++) {
                const a = this.coeffs[i] || new Frac(0);
                const b = poly.coeffs[i] || new Frac(0);
                this.coeffs[i] = a.add(b);
            }
            return this;
        },
        /**
         * Subtracts 2 polynomials
         *
         * @this {Polynomial}
         * @param {Polynomial} poly
         * @returns {Polynomial}
         */
        subtract(poly) {
            const l = Math.max(this.coeffs.length, poly.coeffs.length);
            for (let i = 0; i < l; i++) {
                const a = this.coeffs[i] || new Frac(0);
                const b = poly.coeffs[i] || new Frac(0);
                this.coeffs[i] = a.subtract(b);
            }
            return this;
        },
        /**
         * Divides two polynomials
         *
         * @this {Polynomial}
         * @param {Polynomial} poly
         * @returns {[Polynomial, Polynomial]}
         */
        divide(poly) {
            const { variable } = this;
            /** @type {FracType[]} */
            const dividend = /** @type {FracType[]} */ (core.Utils.arrayClone(this.coeffs));
            /** @type {FracType[]} */
            const divisor = /** @type {FracType[]} */ (core.Utils.arrayClone(poly.coeffs));
            const n = dividend.length;
            const mp = divisor.length - 1;
            /** @type {FracType[]} */
            const quotient = [];

            // Loop through the dividend
            for (let i = 0; i < n; i++) {
                const p = n - (i + 1);
                // Get the difference of the powers
                const d = p - mp;
                // Get the quotient of the coefficients
                const q = dividend[p].divide(divisor[mp]);

                if (d < 0) {
                    break;
                } // The divisor is not greater than the dividend
                // place it in the quotient
                quotient[d] = q;

                for (let j = 0; j <= mp; j++) {
                    // Reduce the dividend
                    dividend[j + d] = dividend[j + d].subtract(divisor[j].multiply(q));
                }
            }

            // Clean up
            const p1 = Polynomial.fromArray(dividend, variable || 'x').trim(); // Pass in x for safety
            const p2 = Polynomial.fromArray(quotient, variable || 'x');
            return [p2, p1];
        },
        /**
         * Multiplies two polynomials
         *
         * @this {Polynomial}
         * @param {Polynomial} poly
         * @returns {Polynomial}
         */
        multiply(poly) {
            const l1 = this.coeffs.length;
            const l2 = poly.coeffs.length;
            /** @type {FracType[]} */
            const c = []; // Array to be returned
            for (let i = 0; i < l1; i++) {
                const x1 = this.coeffs[i];
                for (let j = 0; j < l2; j++) {
                    const k = i + j; // Add the powers together
                    const x2 = poly.coeffs[j];
                    const e = c[k] || new Frac(0); // Get the existing term from the new array
                    c[k] = e.add(x1.multiply(x2)); // Multiply the coefficients and add to new polynomial array
                }
            }
            this.coeffs = c;
            return this;
        },
        /**
         * Checks if a polynomial is zero
         *
         * @this {Polynomial}
         * @returns {boolean}
         */
        isZero() {
            const l = this.coeffs.length;
            for (let i = 0; i < l; i++) {
                const e = this.coeffs[i];
                if (!e.equals(0)) {
                    return false;
                }
            }
            return true;
        },
        /**
         * Substitutes in a number n into the polynomial p(n)
         *
         * @this {Polynomial}
         * @param {number} n
         * @returns {FracType}
         */
        sub(n) {
            let sum = new Frac(0);
            const l = this.coeffs.length;
            for (let i = 0; i < l; i++) {
                const t = this.coeffs[i];
                if (!t.equals(0)) {
                    sum = sum.add(t.multiply(new Frac(n ** i)));
                }
            }
            return sum;
        },
        /**
         * Returns a clone of the polynomial
         *
         * @this {Polynomial}
         * @returns {Polynomial}
         */
        clone() {
            /** @type {Polynomial} */
            const p = new Polynomial();
            p.coeffs = this.coeffs.slice();
            p.variable = this.variable;
            return p;
        },
        /**
         * Gets the degree of the polynomial
         *
         * @this {Polynomial}
         * @returns {number}
         */
        deg() {
            this.trim();
            return this.coeffs.length - 1;
        },
        /**
         * Returns a lead coefficient
         *
         * @this {Polynomial}
         * @returns {FracType}
         */
        lc() {
            return this.coeffs[this.deg()].clone();
        },
        /**
         * Converts polynomial into a monic polynomial
         *
         * @this {Polynomial}
         * @returns {Polynomial}
         */
        monic() {
            const lc = this.lc();
            const l = this.coeffs.length;
            for (let i = 0; i < l; i++) {
                this.coeffs[i] = this.coeffs[i].divide(lc);
            }
            return this;
        },
        /**
         * Returns the GCD of two polynomials
         *
         * @this {Polynomial}
         * @param {Polynomial} poly
         * @returns {Polynomial}
         */
        gcd(poly) {
            // Get the maximum power of each
            const mp1 = this.coeffs.length - 1;
            const mp2 = poly.coeffs.length - 1;
            /** @type {[Polynomial, Polynomial]} */
            let T;
            // Swap so we always have the greater power first
            if (mp1 < mp2) {
                return poly.gcd(this);
            }
            /** @type {Polynomial} */
            let a = this;

            while (!poly.isZero()) {
                const t = poly.clone();
                a = a.clone();
                T = a.divide(t);
                poly = T[1];
                a = t;
            }

            const gcd = core.Math2.QGCD.apply(null, a.coeffs);
            if (!gcd.equals(1)) {
                const l = a.coeffs.length;
                for (let i = 0; i < l; i++) {
                    a.coeffs[i] = a.coeffs[i].divide(gcd);
                }
            }
            return a;
        },
        /**
         * Differentiates the polynomial
         *
         * @this {Polynomial}
         * @returns {Polynomial}
         */
        diff() {
            /** @type {FracType[]} */
            const newArray = [];
            const l = this.coeffs.length;
            for (let i = 1; i < l; i++) {
                newArray.push(this.coeffs[i].multiply(new Frac(i)));
            }
            this.coeffs = newArray;
            return this;
        },
        /**
         * Integrates the polynomial
         *
         * @this {Polynomial}
         * @returns {Polynomial}
         */
        integrate() {
            /** @type {FracType[]} */
            const newArray = [new Frac(0)];
            const l = this.coeffs.length;
            for (let i = 0; i < l; i++) {
                const c = new Frac(i + 1);
                newArray[i + 1] = this.coeffs[i].divide(c);
            }
            this.coeffs = newArray;
            return this;
        },
        /**
         * Returns the Greatest common factor of the polynomial
         *
         * @this {Polynomial}
         * @param {boolean} [toPolynomial] - True if a polynomial is wanted
         * @returns {[FracType, number] | Polynomial}
         */
        gcf(toPolynomial) {
            // Get the first nozero coefficient and returns its power
            /**
             * @param {FracType[]} a
             * @returns {number | undefined}
             */
            const fnz = function (a) {
                for (let i = 0; i < a.length; i++) {
                    if (!a[i].equals(0)) {
                        return i;
                    }
                }
                return undefined;
            };
            /** @type {FracType[]} */
            const ca = [];
            for (let i = 0; i < this.coeffs.length; i++) {
                const c = this.coeffs[i];
                if (!c.equals(0) && ca.indexOf(c) === -1) {
                    ca.push(c);
                }
            }
            /** @type {[FracType, number] | Polynomial} */
            let p = [core.Math2.QGCD.apply(undefined, ca), fnz(this.coeffs) || 0];

            if (toPolynomial) {
                const parr = [];
                parr[p[1] - 1] = p[0];
                p = Polynomial.fromArray(parr, this.variable).fill();
            }

            return p;
        },
        /**
         * Raises a polynomial P to a power p -> P^p. e.g. (x+1)^2
         *
         * @this {Polynomial}
         * @param {boolean} [inclImg] - Include imaginary numbers
         * @returns {number[]}
         */
        quad(inclImg) {
            /** @type {number[]} */
            const roots = [];
            if (this.coeffs.length > 3) {
                throw new Error(`Cannot calculate quadratic order of ${this.coeffs.length - 1}`);
            }
            if (this.coeffs.length === 0) {
                throw new Error('Polynomial array has no terms');
            }
            const a = this.coeffs[2] ? Number(this.coeffs[2].toDecimal()) : 0;
            const b = this.coeffs[1] ? Number(this.coeffs[1].toDecimal()) : 0;
            const c = Number(this.coeffs[0].toDecimal());
            const dsc = b * b - 4 * a * c;
            if (dsc < 0 && !inclImg) {
                return roots;
            }
            roots[0] = (-b + Math.sqrt(dsc)) / (2 * a);
            roots[1] = (-b - Math.sqrt(dsc)) / (2 * a);

            return roots;
        },
        /**
         * Makes polynomial square free
         *
         * @this {Polynomial}
         * @returns {[Polynomial, Polynomial, number]}
         */
        squareFree() {
            const a = this.clone();
            let i = 1;
            const b = a.clone().diff();
            let c = a.clone().gcd(b);
            let w = a.divide(c)[0];
            let output = Polynomial.fromArray([new Frac(1)], a.variable);
            while (!c.equalsNumber(1)) {
                const y = w.gcd(c);
                let z = w.divide(y)[0];
                // One of the factors may have shown up since it's square but smaller than the
                // one where finding
                if (!z.equalsNumber(1) && i > 1) {
                    const t = z.clone();
                    for (let j = 1; j < i; j++) {
                        t.multiply(z.clone());
                    }
                    z = t;
                }
                output = output.multiply(z);
                i++;
                w = y;
                c = c.divide(y)[0];
            }

            return [output, w, i];
        },
        /**
         * Converts polynomial to NerdamerSymbol
         *
         * @this {Polynomial}
         * @returns {NerdamerSymbolType}
         */
        toSymbol() {
            const l = this.coeffs.length;
            const { variable } = this;
            if (l === 0) {
                return new NerdamerSymbol(0);
            }

            // Polynomials must have a variable
            if (!variable) {
                throw new core.exceptions.NerdamerTypeError(
                    'Polynomial.toSymbol requires a variable. Constants should not be converted to Polynomial.'
                );
            }

            const terms = [];

            for (let i = 0; i < l; i++) {
                const e = this.coeffs[i];
                if (!e.equals(0)) {
                    terms.push(`${e}*${variable}^${i}`);
                }
            }
            if (terms.length === 0) {
                return new NerdamerSymbol(0);
            }
            return _.parse(terms.join('+'));
        },
        /**
         * Checks if polynomial is equal to a number
         *
         * @this {Polynomial}
         * @param {number} x
         * @returns {boolean}
         */
        equalsNumber(x) {
            this.trim();
            return this.coeffs.length === 1 && this.coeffs[0].toDecimal() === String(x);
        },
        /**
         * @this {Polynomial}
         * @returns {string}
         */
        toString() {
            return this.toSymbol().toString();
        },
    };

    /**
     * # TODO
     *
     * # THIS METHOD HAS A NASTY HIDDEN BUG. IT HAS INCONSISTENT RETURN TYPES PRIMARILY DUE TO
     *
     * WRONG ASSUMPTIONS AT THE BEGINNING. THE ASSUMPTION WAS THAT COEFFS WERE ALWAYS GOING BE NUMBERS NOT TAKING INTO
     * ACCOUNT THAT IMAGINARY NUMBERS. FIXING THIS BREAKS WAY TOO MANY TESTS AT THEM MOMENT WHICH I DON'T HAVE TO FIX
     *
     * If the symbols is of group PL or CP it will return the multipliers of each symbol as these are polynomial
     * coefficients. CB symbols are glued together by multiplication so the symbol multiplier carries the coefficients
     * for all contained symbols. For S it just returns it's own multiplier. This function doesn't care if it's a
     * polynomial or not
     *
     * @this {NerdamerSymbolType}
     * @param {Array} [c] The coefficient array
     * @param {boolean} [withOrder]
     * @returns {Array}
     */
    NerdamerSymbol.prototype.coeffs = function coeffs(c, withOrder) {
        if (withOrder && !this.isPoly(true)) {
            _.error('Polynomial expected when requesting coefficients with order');
        }
        c ||= [];
        const s = this.clone().distributeMultiplier();
        if (s.isComposite()) {
            for (const x in s.symbols) {
                if (!Object.hasOwn(s.symbols, x)) {
                    continue;
                }
                const sub = s.symbols[x];
                if (sub.isComposite()) {
                    sub.clone().distributeMultiplier().coeffs(c, withOrder);
                } else if (withOrder) {
                    c[sub.isConstant() ? 0 : Number(/** @type {FracType} */ (sub.power).toDecimal())] = sub.multiplier;
                } else {
                    c.push(sub.multiplier);
                }
            }
        } else if (withOrder) {
            c[s.isConstant(true) ? 0 : Number(/** @type {FracType} */ (s.power).toDecimal())] = s.multiplier;
        } else if (s.group === CB && s.isImaginary()) {
            let m = new NerdamerSymbol(s.multiplier);
            s.each(x => {
                // Add the imaginary part
                if (x.isConstant(true) || x.imaginary) {
                    m = /** @type {NerdamerSymbolType} */ (_.multiply(m, x));
                }
            });
            c.push(m);
        } else {
            c.push(s.multiplier);
        }
        // Fill the holes
        if (withOrder) {
            for (let i = 0; i < c.length; i++) {
                if (c[i] === undefined) {
                    c[i] = new NerdamerSymbol(0);
                }
            }
        }
        return c;
    };
    /**
     * @this {NerdamerSymbolType}
     * @param {Record<string, number> & { length: number }} map
     * @returns {MVTerm[]}
     */
    NerdamerSymbol.prototype.tBase = function tBase(map) {
        if (typeof map === 'undefined') {
            throw new Error('NerdamerSymbol.tBase requires a map object!');
        }
        /** @type {MVTerm[]} */
        const terms = [];
        const symbols = /** @type {NerdamerSymbolType[]} */ (this.collectSymbols(null, null, null, true));
        const l = symbols.length;
        for (let i = 0; i < l; i++) {
            const symbol = symbols[i];
            const g = symbol.group;
            /** @type {MVTerm} */
            const nterm = new MVTerm(symbol.multiplier, [], map);
            if (g === CB) {
                for (const x in symbol.symbols) {
                    if (!Object.hasOwn(symbol.symbols, x)) {
                        continue;
                    }
                    const sym = symbol.symbols[x];
                    nterm.terms[map[x]] = /** @type {FracType} */ (sym.power);
                }
            } else {
                nterm.terms[map[symbol.value]] = /** @type {FracType} */ (symbol.power);
            }

            terms.push(nterm.fill());
            nterm.updateCount();
        }
        return terms;
    };
    /**
     * @this {NerdamerSymbolType}
     * @param {string} x
     * @returns {string}
     */
    NerdamerSymbol.prototype.altVar = function altVar(x) {
        const m = this.multiplier.toString();
        const p = this.power.toString();
        return (m === '1' ? '' : `${m}*`) + x + (p === '1' ? '' : `^${p}`);
    };
    /**
     * Checks to see if the symbols contain the same variables
     *
     * @this {NerdamerSymbolType}
     * @param {NerdamerSymbolType} symbol
     * @returns {boolean}
     */
    NerdamerSymbol.prototype.sameVars = function sameVars(symbol) {
        if (!(this.symbols || this.group === symbol.group)) {
            return false;
        }
        for (const x in this.symbols) {
            if (!Object.hasOwn(this.symbols, x)) {
                continue;
            }
            const a = this.symbols[x];
            const b = symbol.symbols[x];
            if (!b) {
                return false;
            }
            if (a.value !== b.value) {
                return false;
            }
        }
        return true;
    };
    /**
     * Groups the terms in a symbol with respect to a variable For instance the symbol {a_b^2_x^2+a_b_x^2+x+6} returns
     * [6,1,a_b+a_b^2]
     *
     * @this {NerdamerSymbolType}
     * @param {string} x
     * @returns {NerdamerSymbolType[]}
     */
    NerdamerSymbol.prototype.groupTerms = function groupTerms(x) {
        x = String(x);
        /** @type {DecomposeResultType | undefined} */
        let f;
        /** @type {number} */
        let p;
        /** @type {NerdamerSymbolType[] | undefined} */
        let egrouped;
        /** @type {NerdamerSymbolType[]} */
        const grouped = [];
        this.each(e => {
            if (e.group === PL) {
                egrouped = e.groupTerms(x);
                for (let i = 0; i < egrouped.length; i++) {
                    const el = egrouped[i];
                    if (el) {
                        grouped[i] = el;
                    }
                }
            } else {
                f = /** @type {DecomposeResultType} */ (core.Utils.decompose_fn(e, x, true));
                p =
                    /** @type {NerdamerSymbolType} */ (f.x).value === x
                        ? Number(/** @type {NerdamerSymbolType} */ (f.x).power)
                        : 0;
                // Check if there's an existing value
                grouped[p] = /** @type {NerdamerSymbolType} */ (_.add(grouped[p] || new NerdamerSymbol(0), f.a));
            }
        });
        return grouped;
    };
    /**
     * Use this to collect Factors
     *
     * @this {NerdamerSymbolType}
     * @returns {NerdamerSymbolType[]}
     */
    NerdamerSymbol.prototype.collectFactors = function collectFactors() {
        /** @type {NerdamerSymbolType[]} */
        const factors = [];
        if (this.group === CB) {
            this.each(x => {
                factors.push(x.clone());
            });
        } else {
            factors.push(this.clone());
        }
        return factors;
    };
    /**
     * A container class for factors
     *
     * @class
     * @this {Factors}
     */
    function Factors() {
        /** @type {Record<string, NerdamerSymbolType>} */
        this.factors = {};
        /** @type {number} */
        this.length = 0;
        /** @type {((s: NerdamerSymbolType) => NerdamerSymbolType) | undefined} */
        this.preAdd = undefined;
        /** @type {number | string | undefined} */
        this.pFactor = undefined;
    }
    /**
     * @this {Factors}
     * @returns {number}
     */
    Factors.prototype.getNumberSymbolics = function getNumberSymbolics() {
        let n = 0;
        this.each(x => {
            if (!x.isConstant(true)) {
                n++;
            }
        });
        return n;
    };
    /**
     * Adds the factors to the factor object
     *
     * @this {Factors}
     * @param {NerdamerSymbolType} s
     * @returns {Factors}
     */
    Factors.prototype.add = function add(s) {
        if (s.equals(0)) {
            return this;
        } // Nothing to add

        // we don't want to carry -1 as a factor. If a factor already exists,
        // then add the minus one to that factor and return.
        if (s.equals(-1) && this.length > 0) {
            const fo = core.Utils.firstObject(this.factors, null, true);
            const newObj = /** @type {NerdamerSymbolType} */ (
                _.symfunction(core.Settings.PARENTHESIS, [fo.obj]).negate()
            );
            delete this.factors[fo.key];
            this.add(newObj);
            this.length--;
            return this;
        }

        if (s.group === CB) {
            const factors = this;
            if (!s.multiplier.equals(1)) {
                factors.add(new NerdamerSymbol(s.multiplier));
            }
            s.each(x => {
                factors.add(x);
            });
        } else {
            if (this.preAdd) // If a preAdd function was defined call it to do prep
            {
                s = this.preAdd(s);
            }
            if (this.pFactor) // If the symbol isn't linear add back the power
            {
                s = /** @type {NerdamerSymbolType} */ (_.pow(s, new NerdamerSymbol(this.pFactor)));
            }

            const isConstant = s.isConstant();
            if (isConstant && s.equals(1)) {
                return this;
            } // Don't add 1
            const v = isConstant ? s.value : s.text();
            if (v in this.factors) {
                this.factors[v] = /** @type {NerdamerSymbolType} */ (_.multiply(this.factors[v], s));
                // Did the addition cancel out the existing factor? If so remove it and decrement the length
                if (this.factors[v].equals(1)) {
                    delete this.factors[v];
                    this.length--;
                }
            } else {
                this.factors[v] = s;
                this.length++;
            }
        }
        return this;
    };
    /**
     * Converts the factor object to a NerdamerSymbol
     *
     * @this {Factors}
     * @returns {NerdamerSymbolType}
     */
    Factors.prototype.toSymbol = function toSymbol() {
        /** @type {NerdamerSymbolType} */
        let factored = new NerdamerSymbol(1);
        const factors = Object.values(this.factors).sort((a, b) => (a.group > b.group ? 1 : -1));

        for (let i = 0, l = factors.length; i < l; i++) {
            const f = factors[i];

            // Don't wrap group S or FN
            const factor =
                f.power.equals(1) && f.fname !== '' /* Don't wrap it twice */
                    ? _.symfunction(core.Settings.PARENTHESIS, [f])
                    : f;

            factored = /** @type {NerdamerSymbolType} */ (_.multiply(factored, factor));
        }
        if (factored.fname === '') {
            factored = NerdamerSymbol.unwrapPARENS(factored);
        }
        return factored;
    };
    /**
     * Merges 2 factor objects into one
     *
     * @this {Factors}
     * @param {Record<string, NerdamerSymbolType>} o
     * @returns {Factors}
     */
    Factors.prototype.merge = function merge(o) {
        for (const x in o) {
            if (x in this.factors) {
                this.factors[x] = /** @type {NerdamerSymbolType} */ (_.multiply(this.factors[x], o[x]));
            } else {
                this.factors[x] = o[x];
            }
        }
        return this;
    };
    /**
     * The iterator for the factor object
     *
     * @this {Factors}
     * @param {(factor: NerdamerSymbolType, key: string) => void} f - Callback
     * @returns {Factors}
     */
    Factors.prototype.each = function each(f) {
        for (const x in this.factors) {
            if (!Object.hasOwn(this.factors, x)) {
                continue;
            }
            let factor = this.factors[x];
            if (factor.fname === core.Settings.PARENTHESIS && factor.isLinear()) {
                factor = factor.args[0];
            }
            f.call(this, factor, x);
        }
        return this;
    };
    /**
     * Return the number of factors contained in the factor object
     *
     * @this {Factors}
     * @returns {number}
     */
    Factors.prototype.count = function count() {
        return keys(this.factors).length;
    };
    /**
     * Cleans up factors from -1
     *
     * @this {Factors}
     * @returns {void}
     */
    Factors.prototype.clean = function clean() {
        try {
            const h = core.Settings.CONST_HASH;
            if (this.factors[h].lessThan(0)) {
                if (this.factors[h].equals(-1)) {
                    delete this.factors[h];
                } else {
                    this.factors[h].negate();
                }
                this.each(x => {
                    x.negate();
                });
            }
        } catch (e) {
            if (/** @type {Error} */ (e).message === 'timeout') {
                throw e;
            }
        }
    };
    /**
     * @this {Factors}
     * @returns {string}
     */
    Factors.prototype.toString = function toString() {
        return this.toSymbol().toString();
    };

    /**
     * A wrapper for performing multivariate division
     *
     * @class
     * @this {MVTerm}
     * @param {FracType} coeff
     * @param {FracType[]} [terms]
     * @param {Record<string, number> & { length: number }} [map]
     */
    function MVTerm(coeff, terms, map) {
        /** @type {FracType[]} */
        this.terms = terms || [];
        /** @type {FracType} */
        this.coeff = coeff;
        /** @type {(Record<string, number> & { length: number }) | undefined} */
        this.map = map; // Careful! all maps are the same object
        /** @type {FracType} */
        this.sum = new Frac(0);
        /** @type {string | undefined} */
        this.image = undefined;
        /** @type {Record<number, string> | undefined} */
        this.revMap = undefined;
        /** @type {number | undefined} */
        this.count = undefined;
    }
    /**
     * @this {MVTerm}
     * @returns {MVTerm}
     */
    MVTerm.prototype.updateCount = function updateCount() {
        this.count ||= 0;
        for (let i = 0; i < this.terms.length; i++) {
            if (!this.terms[i].equals(0)) {
                this.count++;
            }
        }
        return this;
    };
    /**
     * @this {MVTerm}
     * @returns {string}
     */
    MVTerm.prototype.getVars = function getVars() {
        /** @type {string[]} */
        const vars = [];
        for (let i = 0; i < this.terms.length; i++) {
            const term = this.terms[i];
            this.getRevMap();
            if (!term.equals(0) && this.revMap) {
                vars.push(this.revMap[i]);
            }
        }
        return vars.join(' ');
    };
    /**
     * @this {MVTerm}
     * @returns {number}
     */
    MVTerm.prototype.len = function len() {
        if (typeof this.count === 'undefined') {
            this.updateCount();
        }
        return this.count || 0;
    };
    /**
     * @this {MVTerm}
     * @param {Record<number, string>} [revMap]
     * @returns {NerdamerSymbolType}
     */
    MVTerm.prototype.toSymbol = function toSymbol(revMap) {
        revMap ||= this.getRevMap();
        /** @type {NerdamerSymbolType} */
        let symbol = new NerdamerSymbol(this.coeff);
        for (let i = 0; i < this.terms.length; i++) {
            const v = revMap[i];
            const t = this.terms[i];
            if (t.equals(0) || v === CONST_HASH) {
                continue;
            }
            const mapped = new NerdamerSymbol(v);
            mapped.power = t;
            symbol = /** @type {NerdamerSymbolType} */ (_.multiply(symbol, mapped));
        }
        return symbol;
    };
    /**
     * @this {MVTerm}
     * @returns {Record<number, string>}
     */
    MVTerm.prototype.getRevMap = function getRevMap() {
        if (this.revMap) {
            return this.revMap;
        }
        /** @type {Record<number, string>} */
        const o = {};
        if (this.map) {
            for (const x in this.map) {
                if (!Object.hasOwn(this.map, x)) {
                    continue;
                }
                o[this.map[x]] = x;
            }
        }
        this.revMap = o;
        return o;
    };
    /**
     * @this {MVTerm}
     * @returns {MVTerm}
     */
    MVTerm.prototype.generateImage = function generateImage() {
        this.image = this.terms.join(' ');
        return this;
    };
    /**
     * @this {MVTerm}
     * @returns {string}
     */
    MVTerm.prototype.getImg = function getImg() {
        if (!this.image) {
            this.generateImage();
        }
        return this.image || '';
    };
    /**
     * @this {MVTerm}
     * @returns {MVTerm}
     */
    MVTerm.prototype.fill = function fill() {
        const l = this.map ? this.map.length : 0;
        for (let i = 0; i < l; i++) {
            if (typeof this.terms[i] === 'undefined') {
                this.terms[i] = new Frac(0);
            } else {
                this.sum = this.sum.add(this.terms[i]);
            }
        }
        return this;
    };
    /**
     * @this {MVTerm}
     * @param {MVTerm} mvterm
     * @returns {MVTerm}
     */
    MVTerm.prototype.divide = function divide(mvterm) {
        const c = this.coeff.divide(mvterm.coeff);
        const l = this.terms.length;
        /** @type {MVTerm} */
        const newMvterm = new MVTerm(c, [], this.map);
        for (let i = 0; i < l; i++) {
            newMvterm.terms[i] = this.terms[i].subtract(mvterm.terms[i]);
            newMvterm.sum = newMvterm.sum.add(newMvterm.terms[i]);
        }
        return newMvterm;
    };
    /**
     * @this {MVTerm}
     * @param {MVTerm} mvterm
     * @returns {MVTerm}
     */
    MVTerm.prototype.multiply = function multiply(mvterm) {
        const c = this.coeff.multiply(mvterm.coeff);
        const l = this.terms.length;
        /** @type {MVTerm} */
        const newMvterm = new MVTerm(c, [], this.map);
        for (let i = 0; i < l; i++) {
            newMvterm.terms[i] = this.terms[i].add(mvterm.terms[i]);
            newMvterm.sum = newMvterm.sum.add(newMvterm.terms[i]);
        }
        return newMvterm;
    };
    /**
     * @this {MVTerm}
     * @returns {boolean}
     */
    MVTerm.prototype.isZero = function isZero() {
        return this.coeff.equals(0);
    };
    /**
     * @this {MVTerm}
     * @returns {string}
     */
    MVTerm.prototype.toString = function toString() {
        return `{ coeff: ${this.coeff.toString()}, terms: [${this.terms.join(
            ','
        )}]: sum: ${this.sum.toString()}, count: ${this.count}}`;
    };

    /**
     * @param {string[]} arr
     * @returns {Record<string, number> & { length: number }}
     */
    core.Utils.toMapObj = function toMapObj(arr) {
        let c = 0;
        /** @type {Record<string, number> & { length: number }} */
        const o = /** @type {Record<string, number> & { length: number }} */ ({ length: 0 });
        for (let i = 0; i < arr.length; i++) {
            const v = arr[i];
            if (typeof o[v] === 'undefined') {
                o[v] = c;
                c++;
            }
        }
        o.length = c;
        return o;
    };
    /**
     * @template T
     * @param {T} v
     * @param {number} n
     * @param {new (v: T) => T} [Clss]
     * @returns {T[]}
     */
    core.Utils.filledArray = function filledArray(v, n, Clss) {
        const a = [];
        while (n--) {
            a[n] = Clss ? new Clss(v) : v;
        }
        return a;
    };
    /**
     * @param {number[]} arr
     * @returns {number}
     */
    core.Utils.arrSum = function arrSum(arr) {
        let sum = 0;
        const l = arr.length;
        for (let i = 0; i < l; i++) {
            sum += arr[i];
        }
        return sum;
    };
    /**
     * Determines if 2 arrays have intersecting elements.
     *
     * @template T
     * @param {T[]} a
     * @param {T[]} b
     * @returns {boolean} True if a and b have intersecting elements.
     */
    core.Utils.haveIntersection = function haveIntersection(a, b) {
        if (b.length > a.length) {
            [a, b] = [b, a]; // IndexOf to loop over shorter
        }
        return a.some(e => b.indexOf(e) > -1);
    };
    /**
     * Substitutes out functions as variables so they can be used in regular algorithms
     *
     * @param {NerdamerSymbolType} symbol
     * @param {Record<string, string>} [map]
     * @returns {string} The expression string
     */
    core.Utils.subFunctions = function subFunctions(symbol, map) {
        map ||= {};
        /** @type {string[]} */
        const subbed = [];
        const vars = new Set(variables(symbol));
        symbol.each(x => {
            if (x.group === FN || x.previousGroup === FN) {
                // We need a new variable name so why not use one of the existing
                const val = core.Utils.text(x, 'hash');
                const tvar = map[val];
                if (tvar) {
                    subbed.push(x.altVar(tvar));
                } else {
                    // Generate a unique enough name
                    // GM make sure it's not the name of an existing variable
                    let i = 0;
                    let t;
                    do {
                        t = x.fname + keys(map).length + (i > 0 ? String(i) : '');
                        i++;
                    } while (vars.has(t));
                    map[val] = t;
                    subbed.push(x.altVar(t));
                }
            } else if (x.group === CB || x.group === PL || x.group === CP) {
                subbed.push(core.Utils.subFunctions(x, map));
            } else {
                subbed.push(x.text());
            }
        });
        if (symbol.group === CP || symbol.group === PL) {
            return symbol.altVar(core.Utils.inBrackets(subbed.join('+')));
        }
        if (symbol.group === CB) {
            return symbol.altVar(core.Utils.inBrackets(subbed.join('*')));
        }
        return symbol.text();
    };
    /**
     * @param {Record<string, string>} map
     * @returns {Record<string, NerdamerSymbolType>}
     */
    core.Utils.getFunctionsSubs = function getFunctionsSubs(map) {
        /** @type {Record<string, NerdamerSymbolType>} */
        const subs = {};
        // Prepare substitutions
        for (const x in map) {
            if (!Object.hasOwn(map, x)) {
                continue;
            }
            subs[map[x]] = _.parse(x);
        }
        return subs;
    };

    /** @type {AlgebraModuleType} */
    const __ = (core.Algebra = {
        version: '1.4.6',
        /**
         * @param {NerdamerSymbolType | Array} symbol
         * @param {number} [decp]
         * @returns {(string | number)[]}
         */
        proots(symbol, decp) {
            // The roots will be rounded up to 7 decimal places.
            // if this causes trouble you can explicitly pass in a different number of places
            // rarr for polynomial of power n is of format [n, coeff x^n, coeff x^(n-1), ..., coeff x^0]
            decp ||= 7;
            const zeros = 0;
            /** @type {(string | number)[]} */
            const knownRoots = [];
            /**
             * @param {FracType[]} rarr
             * @param {(string | number)[]} powers
             * @param {number} max
             * @returns {(string | number)[]}
             */
            const getRoots = function (rarr, powers, max) {
                const roots = calcroots(rarr, powers, max).concat(knownRoots);
                for (let i = 0; i < zeros; i++) {
                    roots.unshift(0);
                }
                return /** @type {string[]} */ (roots);
            };

            if (core.Utils.isSymbol(symbol) && /** @type {NerdamerSymbolType} */ (symbol).isPoly()) {
                let sym = /** @type {NerdamerSymbolType} */ (symbol);
                sym.distributeMultiplier();
                // Make it so the symbol has a constants as the lowest term
                if (sym.group === PL) {
                    const lowestPow = core.Utils.arrayMin(
                        /** @type {number[]} */ (/** @type {unknown} */ (keys(sym.symbols)))
                    );
                    const lowestSymbol = sym.symbols[lowestPow].clone().toUnitMultiplier();
                    sym = /** @type {NerdamerSymbolType} */ (_.expand(_.divide(sym, lowestSymbol)));
                    knownRoots.push(0); // Add zero since this is a known root
                }
                if (sym.group === core.groups.S) {
                    return [/** @type {string} */ ('0')];
                }
                if (sym.group === core.groups.PL) {
                    const powers = keys(sym.symbols);
                    const minpower = core.Utils.arrayMin(/** @type {number[]} */ (/** @type {unknown} */ (powers)));
                    sym = /** @type {NerdamerSymbolType} */ (
                        core.PARSER.divide(sym, core.PARSER.parse(`${sym.value}^${minpower}`))
                    );
                }

                const variable = keys(sym.symbols).sort().pop();
                const subSym = sym.group === core.groups.PL ? sym.symbols : sym.symbols[variable || ''];
                const g = subSym.group;
                const powers = g === S ? [/** @type {FracType} */ (subSym.power).toDecimal()] : keys(subSym.symbols);
                /** @type {(FracType | number)[]} */
                const rarr = [];
                const max = core.Utils.arrayMax(/** @type {number[]} */ (/** @type {unknown} */ (powers))); // Maximum power and degree of polynomial to be solved

                // Prepare the data
                for (let i = 1; i <= max; i++) {
                    /** @type {FracType | number} */
                    let c = 0; // If there is no power then the hole must be filled with a zero
                    if (powers.indexOf(`${i}`) !== -1) {
                        if (g === S) {
                            c = /** @type {FracType} */ (subSym.multiplier);
                        } else {
                            c = /** @type {FracType} */ (subSym.symbols[i].multiplier);
                        }
                    }
                    // Insert the coeffient but from the front
                    rarr.unshift(c);
                }

                rarr.push(/** @type {NerdamerSymbolType} */ (symbol).symbols[CONST_HASH].multiplier);

                if (sym.group === S) {
                    rarr[0] = sym.multiplier;
                } // The symbol maybe of group CP with one variable

                return /** @type {(string | number)[]} */ (getRoots(/** @type {FracType[]} */ (rarr), powers, max));
            }
            if (core.Utils.isArray(symbol)) {
                const parr = symbol;
                const rarr = [];
                const powers = [];
                let lastPower = 0;
                for (let i = 0; i < parr.length; i++) {
                    const coeff = parr[i][0];
                    const pow = parr[i][1];
                    const d = pow - lastPower - 1;
                    // Insert the zeros
                    for (let j = 0; j < d; j++) {
                        rarr.unshift(0);
                    }

                    rarr.unshift(coeff);
                    if (pow !== 0) {
                        powers.push(pow);
                    }
                    lastPower = pow;
                }
                const max = Math.max.apply(undefined, powers);

                return getRoots(rarr, powers, max);
            }
            throw new core.exceptions.NerdamerTypeError('Cannot calculate roots. NerdamerSymbol must be a polynomial!');

            function calcroots(coeffArr, powArr, maxPow) {
                const MAXDEGREE = 100; // Degree of largest polynomial accepted by this script.
                let i;

                // Make a clone of the coefficients before appending the max power
                const p = coeffArr.slice(0);

                // Divide the string up into its individual entries, which--presumably--are separated by whitespace
                coeffArr.unshift(maxPow);

                if (maxPow > MAXDEGREE) {
                    throw new core.exceptions.ValueLimitExceededError(
                        `This utility accepts polynomials of degree up to ${MAXDEGREE}. `
                    );
                }

                const zeroi = []; // Vector of imaginary components of roots
                const degreePar = {}; // DegreePar is a dummy variable for passing the parameter POLYDEGREE by reference
                degreePar.Degree = maxPow;

                for (i = 0; i < maxPow; i++) {
                    zeroi.push(0);
                }
                const zeror = zeroi.slice(0); // Vector of real components of roots

                // Find the roots
                // --> Begin Jenkins-Traub

                /*
                 * A verbatim copy of Mr. David Binner's Jenkins-Traub port
                 */
                function quadSdAk1(NN, u, v, poly, q, iPar) {
                    // Divides poly by the quadratic 1, u, v placing the quotient in q and the remainder in a, b
                    // iPar is a dummy variable for passing in the two parameters--a and b--by reference
                    q[0] = iPar.b = poly[0];
                    q[1] = iPar.a = -(u * iPar.b) + poly[1];

                    for (let idx = 2; idx < NN; idx++) {
                        q[idx] = -(u * iPar.a + v * iPar.b) + poly[idx];
                        iPar.b = iPar.a;
                        iPar.a = q[idx];
                    }
                }

                function calcScAk1(DBL_EPSILON, degree, a, b, iPar, K, u, v, qk) {
                    // This routine calculates scalar quantities used to compute the next K polynomial and
                    // new estimates of the quadratic coefficients.
                    // calcSC -	integer variable set here indicating how the calculations are normalized
                    // to avoid overflow.
                    // iPar is a dummy variable for passing in the nine parameters--a1, a3, a7, c, d, e, f, g, and h --by reference

                    // sdPar is a dummy variable for passing the two parameters--c and d--into quadSdAk1 by reference
                    const sdPar = {};
                    // TYPE = 3 indicates the quadratic is almost a factor of K
                    let dumFlag = 3;

                    // Synthetic division of K by the quadratic 1, u, v
                    sdPar.b = sdPar.a = 0.0;
                    quadSdAk1(degree, u, v, K, qk, sdPar);
                    iPar.c = sdPar.a;
                    iPar.d = sdPar.b;

                    if (Math.abs(iPar.c) <= 100.0 * DBL_EPSILON * Math.abs(K[degree - 1])) {
                        if (Math.abs(iPar.d) <= 100.0 * DBL_EPSILON * Math.abs(K[degree - 2])) {
                            return dumFlag;
                        }
                    }

                    iPar.h = v * b;
                    if (Math.abs(iPar.d) >= Math.abs(iPar.c)) {
                        // TYPE = 2 indicates that all formulas are divided by d
                        dumFlag = 2;
                        iPar.e = a / iPar.d;
                        iPar.f = iPar.c / iPar.d;
                        iPar.g = u * b;
                        iPar.a3 = iPar.e * (iPar.g + a) + iPar.h * (b / iPar.d);
                        iPar.a1 = -a + iPar.f * b;
                        iPar.a7 = iPar.h + (iPar.f + u) * a;
                    } else {
                        // TYPE = 1 indicates that all formulas are divided by c;
                        dumFlag = 1;
                        iPar.e = a / iPar.c;
                        iPar.f = iPar.d / iPar.c;
                        iPar.g = iPar.e * u;
                        iPar.a3 = iPar.e * a + (iPar.g + iPar.h / iPar.c) * b;
                        iPar.a1 = -(a * (iPar.d / iPar.c)) + b;
                        iPar.a7 = iPar.g * iPar.d + iPar.h * iPar.f + a;
                    }
                    return dumFlag;
                }

                function nextKAk1(DBL_EPSILON, degree, tFlag, a, b, iPar, K, qk, qp) {
                    // Computes the next K polynomials using the scalars computed in calcScAk1
                    // iPar is a dummy variable for passing in three parameters--a1, a3, and a7
                    if (tFlag === 3) {
                        // Use unscaled form of the recurrence
                        K[1] = K[0] = 0.0;
                        for (let idx = 2; idx < degree; idx++) {
                            K[idx] = qk[idx - 2];
                        }
                        return;
                    }

                    const temp = tFlag === 1 ? b : a;
                    if (Math.abs(iPar.a1) > 10.0 * DBL_EPSILON * Math.abs(temp)) {
                        // Use scaled form of the recurrence
                        iPar.a7 /= iPar.a1;
                        iPar.a3 /= iPar.a1;
                        K[0] = qp[0];
                        K[1] = -(qp[0] * iPar.a7) + qp[1];
                        for (let idx = 2; idx < degree; idx++) {
                            K[idx] = -(qp[idx - 1] * iPar.a7) + qk[idx - 2] * iPar.a3 + qp[idx];
                        }
                    } else {
                        // If a1 is nearly zero, then use a special form of the recurrence
                        K[0] = 0.0;
                        K[1] = -(qp[0] * iPar.a7);
                        for (let idx = 2; idx < degree; idx++) {
                            K[idx] = -(qp[idx - 1] * iPar.a7) + qk[idx - 2] * iPar.a3;
                        }
                    }
                }

                function newestAk1(tFlag, iPar, a, a1, a3, a7, b, c, d, f, g, h, u, v, K, degree, poly) {
                    // Compute new estimates of the quadratic coefficients using the scalars computed in calcScAk1
                    // iPar is a dummy variable for passing in the two parameters--uu and vv--by reference
                    // iPar.a = uu, iPar.b = vv

                    let a4;
                    let a5;
                    let b1;
                    let b2;
                    let c1;
                    let c2;
                    let c3;
                    let c4;
                    let temp;
                    iPar.b = iPar.a = 0.0; // The quadratic is zeroed

                    if (tFlag === 3) {
                        // No action needed when tFlag is 3
                    } else {
                        if (tFlag === 2) {
                            a4 = (a + g) * f + h;
                            a5 = (f + u) * c + v * d;
                        } else {
                            a4 = a + u * b + h * f;
                            a5 = c + (u + v * f) * d;
                        }

                        // Evaluate new quadratic coefficients
                        b1 = -(K[degree - 1] / poly[degree]);
                        b2 = -(K[degree - 2] + b1 * poly[degree - 1]) / poly[degree];
                        c1 = v * b2 * a1;
                        c2 = b1 * a7;
                        c3 = b1 * b1 * a3;
                        c4 = -(c2 + c3) + c1;
                        temp = -c4 + a5 + b1 * a4;
                        if (temp !== 0.0) {
                            iPar.a = -((u * (c3 + c2) + v * (b1 * a1 + b2 * a7)) / temp) + u;
                            iPar.b = v * (1.0 + c4 / temp);
                        }
                    }
                }

                function quadAk1(a, b1, c, iPar) {
                    // Calculates the zeros of the quadratic a*Z^2 + b1*Z + c
                    // The quadratic formula, modified to avoid overflow, is used to find the larger zero if the
                    // zeros are real and both zeros are complex. The smaller real zero is found directly from
                    // the product of the zeros c/a.

                    // iPar is a dummy variable for passing in the four parameters--sr, si, lr, and li--by reference

                    let d;
                    let e;
                    iPar.sr = iPar.si = iPar.lr = iPar.li = 0.0;

                    if (a === 0) {
                        iPar.sr = b1 === 0 ? iPar.sr : -(c / b1);
                        return;
                    }
                    if (c === 0) {
                        iPar.lr = -(b1 / a);
                        return;
                    }

                    // Compute discriminant avoiding overflow
                    const b = b1 / 2.0;
                    if (Math.abs(b) < Math.abs(c)) {
                        e = c >= 0 ? a : -a;
                        e = -e + b * (b / Math.abs(c));
                        d = Math.sqrt(Math.abs(e)) * Math.sqrt(Math.abs(c));
                    } else {
                        e = -((a / b) * (c / b)) + 1.0;
                        d = Math.sqrt(Math.abs(e)) * Math.abs(b);
                    }

                    if (e >= 0) {
                        // Real zeros
                        d = b >= 0 ? -d : d;
                        iPar.lr = (-b + d) / a;
                        iPar.sr = iPar.lr === 0 ? iPar.sr : c / iPar.lr / a;
                    } else {
                        // Complex conjugate zeros
                        iPar.lr = iPar.sr = -(b / a);
                        iPar.si = Math.abs(d / a);
                        iPar.li = -iPar.si;
                    }
                }

                function quadItAk1(DBL_EPSILON, degree, iPar, uu, vv, qp, NN, sdPar, poly, qk, calcPar, K) {
                    // Variable-shift K-polynomial iteration for a quadratic factor converges only if the
                    // zeros are equimodular or nearly so.
                    // iPar is a dummy variable for passing in the five parameters--NZ, lzi, lzr, szi, and szr--by reference
                    // sdPar is a dummy variable for passing the two parameters--a and b--in by reference
                    // calcPar is a dummy variable for passing the nine parameters--a1, a3, a7, c, d, e, f, g, and h --in by reference

                    // qPar is a dummy variable for passing the four parameters--szr, szi, lzr, and lzi--into quadAk1 by reference
                    const qPar = {};
                    let ee;
                    let mp;
                    let omp;
                    /** @type {number} */
                    let relstp = 0;
                    let t;
                    let u;
                    let ui;
                    let v;
                    let vi;
                    let zm;
                    let idx;
                    let j = 0;
                    let tFlag;
                    let triedFlag = 0; // Integer variables

                    iPar.NZ = 0; // Number of zeros found
                    u = uu; // Uu and vv are coefficients of the starting quadratic
                    v = vv;

                    do {
                        qPar.li = qPar.lr = qPar.si = qPar.sr = 0.0;
                        quadAk1(1.0, u, v, qPar);
                        iPar.szr = qPar.sr;
                        iPar.szi = qPar.si;
                        iPar.lzr = qPar.lr;
                        iPar.lzi = qPar.li;

                        // Return if roots of the quadratic are real and not close to multiple or nearly
                        // equal and of opposite sign.
                        if (Math.abs(Math.abs(iPar.szr) - Math.abs(iPar.lzr)) > 0.01 * Math.abs(iPar.lzr)) {
                            break;
                        }

                        // Evaluate polynomial by quadratic synthetic division

                        quadSdAk1(NN, u, v, poly, qp, sdPar);

                        mp = Math.abs(-(iPar.szr * sdPar.b) + sdPar.a) + Math.abs(iPar.szi * sdPar.b);

                        // Compute a rigorous bound on the rounding error in evaluating p

                        zm = Math.sqrt(Math.abs(v));
                        ee = 2.0 * Math.abs(qp[0]);
                        t = -(iPar.szr * sdPar.b);

                        for (idx = 1; idx < degree; idx++) {
                            ee = ee * zm + Math.abs(qp[idx]);
                        }

                        ee = ee * zm + Math.abs(t + sdPar.a);
                        ee =
                            (9.0 * ee + 2.0 * Math.abs(t) - 7.0 * (Math.abs(sdPar.a + t) + zm * Math.abs(sdPar.b))) *
                            DBL_EPSILON;

                        // Iteration has converged sufficiently if the polynomial value is less than 20 times this bound
                        if (mp <= 20.0 * ee) {
                            iPar.NZ = 2;
                            break;
                        }

                        j++;
                        // Stop iteration after 20 steps
                        if (j > 20) {
                            break;
                        }
                        if (j >= 2) {
                            if (relstp <= 0.01 && mp >= omp && !triedFlag) {
                                // A cluster appears to be stalling the convergence. Five fixed shift
                                // steps are taken with a u, v close to the cluster.
                                relstp = relstp < DBL_EPSILON ? Math.sqrt(DBL_EPSILON) : Math.sqrt(relstp);
                                u -= u * relstp;
                                v += v * relstp;

                                quadSdAk1(NN, u, v, poly, qp, sdPar);
                                for (idx = 0; idx < 5; idx++) {
                                    tFlag = calcScAk1(DBL_EPSILON, degree, sdPar.a, sdPar.b, calcPar, K, u, v, qk);
                                    nextKAk1(DBL_EPSILON, degree, tFlag, sdPar.a, sdPar.b, calcPar, K, qk, qp);
                                }

                                triedFlag = 1;
                                j = 0;
                            }
                        }
                        omp = mp;

                        // Calculate next K polynomial and new u and v
                        tFlag = calcScAk1(DBL_EPSILON, degree, sdPar.a, sdPar.b, calcPar, K, u, v, qk);
                        nextKAk1(DBL_EPSILON, degree, tFlag, sdPar.a, sdPar.b, calcPar, K, qk, qp);
                        tFlag = calcScAk1(DBL_EPSILON, degree, sdPar.a, sdPar.b, calcPar, K, u, v, qk);
                        newestAk1(
                            tFlag,
                            sdPar,
                            sdPar.a,
                            calcPar.a1,
                            calcPar.a3,
                            calcPar.a7,
                            sdPar.b,
                            calcPar.c,
                            calcPar.d,
                            calcPar.f,
                            calcPar.g,
                            calcPar.h,
                            u,
                            v,
                            K,
                            degree,
                            poly
                        );
                        ui = sdPar.a;
                        vi = sdPar.b;

                        // If vi is zero, the iteration is not converging
                        if (vi !== 0) {
                            relstp = Math.abs((-v + vi) / vi);
                            u = ui;
                            v = vi;
                        }
                    } while (vi !== 0);
                }

                function realItAk1(DBL_EPSILON, iPar, sdPar, degree, poly, NN, qp, K, qk) {
                    // Variable-shift H-polynomial iteration for a real zero
                    // sss	- starting iterate = sdPar.a
                    // NZ		- number of zeros found = iPar.NZ
                    // dumFlag	- flag to indicate a pair of zeros near real axis, returned to iFlag

                    let ee;
                    let kv;
                    let mp;
                    let ms;
                    let omp;
                    let pv;
                    let s;
                    let t;
                    let dumFlag;
                    let idx;
                    let j;
                    const nm1 = degree - 1; // Integer variables

                    iPar.NZ = j = dumFlag = 0;
                    s = sdPar.a;

                    for (;;) {
                        pv = poly[0];

                        // Evaluate p at s
                        qp[0] = pv;
                        for (idx = 1; idx < NN; idx++) {
                            qp[idx] = pv = pv * s + poly[idx];
                        }
                        mp = Math.abs(pv);

                        // Compute a rigorous bound on the error in evaluating p
                        ms = Math.abs(s);
                        ee = 0.5 * Math.abs(qp[0]);
                        for (idx = 1; idx < NN; idx++) {
                            ee = ee * ms + Math.abs(qp[idx]);
                        }

                        // Iteration has converged sufficiently if the polynomial value is less than
                        // 20 times this bound
                        if (mp <= 20.0 * DBL_EPSILON * (2.0 * ee - mp)) {
                            iPar.NZ = 1;
                            iPar.szr = s;
                            iPar.szi = 0.0;
                            break;
                        }
                        j++;
                        // Stop iteration after 10 steps
                        if (j > 10) {
                            break;
                        }

                        if (j >= 2) {
                            if (Math.abs(t) <= 0.001 * Math.abs(-t + s) && mp > omp) {
                                // A cluster of zeros near the real axis has been encountered.
                                // Return with iFlag set to initiate a quadratic iteration.
                                dumFlag = 1;
                                iPar.a = s;
                                break;
                            } // End if ((fabs(t) <= 0.001*fabs(s - t)) && (mp > omp))
                        } // End if (j >= 2)

                        // Return if the polynomial value has increased significantly
                        omp = mp;

                        // Compute t, the next polynomial and the new iterate
                        qk[0] = kv = K[0];
                        for (idx = 1; idx < degree; idx++) {
                            qk[idx] = kv = kv * s + K[idx];
                        }

                        if (Math.abs(kv) > Math.abs(K[nm1]) * 10.0 * DBL_EPSILON) {
                            // Use the scaled form of the recurrence if the value of K at s is non-zero
                            t = -(pv / kv);
                            K[0] = qp[0];
                            for (idx = 1; idx < degree; idx++) {
                                K[idx] = t * qk[idx - 1] + qp[idx];
                            }
                        } else {
                            // Use unscaled form
                            K[0] = 0.0;
                            for (idx = 1; idx < degree; idx++) {
                                K[idx] = qk[idx - 1];
                            }
                        }

                        kv = K[0];
                        for (idx = 1; idx < degree; idx++) {
                            kv = kv * s + K[idx];
                        }
                        t = Math.abs(kv) > Math.abs(K[nm1]) * 10.0 * DBL_EPSILON ? -(pv / kv) : 0.0;
                        s += t;
                    }
                    return dumFlag;
                }

                function fxshfrAk1(DBL_EPSILON, MDP1, L2, sr, v, K, degree, poly, NN, qp, u, iPar) {
                    // Computes up to L2 fixed shift K-polynomials, testing for convergence in the linear or
                    // quadratic case. Initiates one of the variable shift iterations and returns with the
                    // number of zeros found.
                    // L2	limit of fixed shift steps
                    // iPar is a dummy variable for passing in the five parameters--NZ, lzi, lzr, szi, and szr--by reference
                    // NZ	number of zeros found
                    const sdPar = {}; // SdPar is a dummy variable for passing the two parameters--a and b--into quadSdAk1 by reference
                    const calcPar = {};
                    // CalcPar is a dummy variable for passing the nine parameters--a1, a3, a7, c, d, e, f, g, and h --into calcScAk1 by reference

                    const qk = new Array(MDP1);
                    const svk = new Array(MDP1);
                    let a;
                    let b;
                    let betas;
                    let betav;
                    let oss;
                    let ots;
                    let otv;
                    let ovv;
                    let s;
                    let ss;
                    let ts;
                    let tss;
                    let tv;
                    let tvv;
                    let ui;
                    let vi;
                    let vv;
                    let fflag;
                    let idx;
                    let iFlag = 1;
                    let j;
                    let spass;
                    let stry;
                    let tFlag;
                    let vpass;
                    let vtry; // Integer variables

                    iPar.NZ = 0;
                    betav = betas = 0.25;
                    oss = sr;
                    ovv = v;

                    // Evaluate polynomial by synthetic division
                    sdPar.b = sdPar.a = 0.0;
                    quadSdAk1(NN, u, v, poly, qp, sdPar);
                    a = sdPar.a;
                    b = sdPar.b;
                    calcPar.h =
                        calcPar.g =
                        calcPar.f =
                        calcPar.e =
                        calcPar.d =
                        calcPar.c =
                        calcPar.a7 =
                        calcPar.a3 =
                        calcPar.a1 =
                            0.0;
                    tFlag = calcScAk1(DBL_EPSILON, degree, a, b, calcPar, K, u, v, qk);

                    for (j = 0; j < L2; j++) {
                        fflag = 1;

                        // Calculate next K polynomial and estimate v
                        nextKAk1(DBL_EPSILON, degree, tFlag, a, b, calcPar, K, qk, qp);
                        tFlag = calcScAk1(DBL_EPSILON, degree, a, b, calcPar, K, u, v, qk);

                        // Use sdPar for passing in uu and vv instead of defining a brand-new variable.
                        // sdPar.a = ui, sdPar.b = vi
                        newestAk1(
                            tFlag,
                            sdPar,
                            a,
                            calcPar.a1,
                            calcPar.a3,
                            calcPar.a7,
                            b,
                            calcPar.c,
                            calcPar.d,
                            calcPar.f,
                            calcPar.g,
                            calcPar.h,
                            u,
                            v,
                            K,
                            degree,
                            poly
                        );
                        ui = sdPar.a;
                        vv = vi = sdPar.b;

                        // Estimate s
                        ss = K[degree - 1] === 0.0 ? 0.0 : -(poly[degree] / K[degree - 1]);
                        ts = tv = 1.0;

                        if (j !== 0 && tFlag !== 3) {
                            // Compute relative measures of convergence of s and v sequences
                            tv = vv === 0.0 ? tv : Math.abs((vv - ovv) / vv);
                            ts = ss === 0.0 ? ts : Math.abs((ss - oss) / ss);

                            // If decreasing, multiply the two most recent convergence measures
                            tvv = tv < otv ? tv * otv : 1.0;
                            tss = ts < ots ? ts * ots : 1.0;

                            // Compare with convergence criteria
                            vpass = tvv < betav ? 1 : 0;
                            spass = tss < betas ? 1 : 0;

                            if (spass || vpass) {
                                // At least one sequence has passed the convergence test.
                                // Store variables before iterating

                                for (idx = 0; idx < degree; idx++) {
                                    svk[idx] = K[idx];
                                }
                                s = ss;

                                // Choose iteration according to the fastest converging sequence

                                stry = vtry = 0;

                                for (;;) {
                                    if (fflag && (fflag = 0) === 0 && spass && (!vpass || tss < tvv)) {
                                        // Do nothing. Provides a quick "short circuit".
                                    } else {
                                        quadItAk1(
                                            DBL_EPSILON,
                                            degree,
                                            iPar,
                                            ui,
                                            vi,
                                            qp,
                                            NN,
                                            sdPar,
                                            poly,
                                            qk,
                                            calcPar,
                                            K
                                        );
                                        a = sdPar.a;
                                        b = sdPar.b;

                                        if (iPar.NZ > 0) {
                                            return;
                                        }

                                        // Quadratic iteration has failed. Flag that it has been tried and decrease the
                                        // convergence criterion
                                        iFlag = vtry = 1;
                                        betav *= 0.25;

                                        // Try linear iteration if it has not been tried and the s sequence is converging
                                        if (stry || !spass) {
                                            iFlag = 0;
                                        } else {
                                            for (idx = 0; idx < degree; idx++) {
                                                K[idx] = svk[idx];
                                            }
                                        }
                                    }
                                    // Fflag = 0;
                                    if (iFlag !== 0) {
                                        // Use sdPar for passing in s instead of defining a brand-new variable.
                                        // sdPar.a = s
                                        sdPar.a = s;
                                        iFlag = realItAk1(DBL_EPSILON, iPar, sdPar, degree, poly, NN, qp, K, qk);
                                        s = sdPar.a;

                                        if (iPar.NZ > 0) {
                                            return;
                                        }

                                        // Linear iteration has failed. Flag that it has been tried and decrease the
                                        // convergence criterion
                                        stry = 1;
                                        betas *= 0.25;

                                        if (iFlag !== 0) {
                                            // If linear iteration signals an almost double real zero, attempt quadratic iteration
                                            ui = -(s + s);
                                            vi = s * s;
                                            continue;
                                        }
                                    }

                                    // Restore variables
                                    for (idx = 0; idx < degree; idx++) {
                                        K[idx] = svk[idx];
                                    }

                                    // Try quadratic iteration if it has not been tried and the v sequence is converging
                                    if (!vpass || vtry) {
                                        break;
                                    } // Break out of infinite for loop
                                }

                                // Re-compute qp and scalar values to continue the second stage

                                quadSdAk1(NN, u, v, poly, qp, sdPar);
                                a = sdPar.a;
                                b = sdPar.b;

                                tFlag = calcScAk1(DBL_EPSILON, degree, a, b, calcPar, K, u, v, qk);
                            }
                        }
                        ovv = vv;
                        oss = ss;
                        otv = tv;
                        ots = ts;
                    }
                }

                function rpSolve(degPar, poly, zeroReal, zeroImag) {
                    let degree = degPar.Degree;
                    const RADFAC = Math.PI / 180; // Degrees-to-radians conversion factor = PI/180
                    const LB2 = Math.LN2; // Dummy variable to avoid re-calculating this value in loop below
                    const MDP1 = degPar.Degree + 1;
                    const K = new Array(MDP1);
                    const pt = new Array(MDP1);
                    const qp = new Array(MDP1);
                    const temp = new Array(MDP1);
                    // QPar is a dummy variable for passing the four parameters--sr, si, lr, and li--by reference
                    const qPar = {};
                    // FxshfrPar is a dummy variable for passing parameters by reference : NZ, lzi, lzr, szi, szr);
                    const fxshfrPar = {};
                    let bnd;
                    let DBL_EPSILON;
                    let df;
                    let dx;
                    let factor;
                    let ff;
                    let moduliMax;
                    let moduliMin;
                    let sc;
                    let x;
                    let xm;
                    let aa;
                    let bb;
                    let cc;
                    let sr;
                    let t;
                    let u;
                    let xxx;
                    let j;
                    let jj;
                    let l;
                    let NM1;
                    let NN;
                    let zerok; // Integer variables

                    // Calculate the machine epsilon and store in the variable DBL_EPSILON.
                    // To calculate this value, just use existing variables rather than create new ones that will be used only for this code block
                    aa = 1.0;
                    do {
                        DBL_EPSILON = aa;
                        aa /= 2;
                        bb = 1.0 + aa;
                    } while (bb > 1.0);

                    const LO = Number.MIN_VALUE / DBL_EPSILON;
                    const cosr = Math.cos(94.0 * RADFAC); // = -0.069756474
                    const sinr = Math.sin(94.0 * RADFAC); // = 0.99756405
                    let xx = Math.sqrt(0.5); // = 0.70710678
                    let yy = -xx;

                    fxshfrPar.NZ = j = 0;
                    fxshfrPar.szr = fxshfrPar.szi = fxshfrPar.lzr = fxshfrPar.lzi = 0.0;

                    // Remove zeros at the origin, if any
                    while (poly[degree] === 0) {
                        zeroReal[j] = zeroImag[j] = 0;
                        degree--;
                        j++;
                    }
                    NN = degree + 1;

                    // >>>>> Begin Main Loop <<<<<
                    while (degree >= 1) {
                        // Main loop
                        // Start the algorithm for one zero
                        if (degree <= 2) {
                            // Calculate the final zero or pair of zeros
                            if (degree < 2) {
                                zeroReal[degPar.Degree - 1] = -(poly[1] / poly[0]);
                                zeroImag[degPar.Degree - 1] = 0;
                            } else {
                                qPar.li = qPar.lr = qPar.si = qPar.sr = 0.0;
                                quadAk1(poly[0], poly[1], poly[2], qPar);
                                zeroReal[degPar.Degree - 2] = qPar.sr;
                                zeroImag[degPar.Degree - 2] = qPar.si;
                                zeroReal[degPar.Degree - 1] = qPar.lr;
                                zeroImag[degPar.Degree - 1] = qPar.li;
                            }
                            break;
                        }

                        // Find the largest and smallest moduli of the coefficients
                        moduliMax = 0.0;
                        moduliMin = Number.MAX_VALUE;

                        for (i = 0; i < NN; i++) {
                            x = Math.abs(poly[i]);
                            if (x > moduliMax) {
                                moduliMax = x;
                            }
                            if (x !== 0 && x < moduliMin) {
                                moduliMin = x;
                            }
                        }

                        // Scale if there are large or very small coefficients
                        // Computes a scale factor to multiply the coefficients of the polynomial. The scaling
                        // is done to avoid overflow and to avoid undetected underflow interfering with the
                        // convergence criterion.
                        // The factor is a power of the base.
                        sc = LO / moduliMin;

                        if ((sc <= 1.0 && moduliMax >= 10) || (sc > 1.0 && Number.MAX_VALUE / sc >= moduliMax)) {
                            sc = sc === 0 ? Number.MIN_VALUE : sc;
                            l = Math.floor(Math.log(sc) / LB2 + 0.5);
                            factor = 2.0 ** l;
                            if (factor !== 1.0) {
                                for (i = 0; i < NN; i++) {
                                    poly[i] *= factor;
                                }
                            }
                        }

                        // Compute lower bound on moduli of zeros
                        for (let idx = 0; idx < NN; idx++) {
                            pt[idx] = Math.abs(poly[idx]);
                        }
                        pt[degree] = -pt[degree];
                        NM1 = degree - 1;

                        // Compute upper estimate of bound
                        x = Math.exp((Math.log(-pt[degree]) - Math.log(pt[0])) / degree);

                        if (pt[NM1] !== 0) {
                            // If Newton step at the origin is better, use it
                            xm = -pt[degree] / pt[NM1];
                            x = xm < x ? xm : x;
                        }

                        // Chop the interval (0, x) until ff <= 0
                        xm = x;
                        do {
                            x = xm;
                            xm = 0.1 * x;
                            ff = pt[0];
                            for (let idx = 1; idx < NN; idx++) {
                                ff = ff * xm + pt[idx];
                            }
                        } while (ff > 0); // End do-while loop

                        dx = x;
                        // Do Newton iteration until x converges to two decimal places

                        do {
                            df = ff = pt[0];
                            for (let idx = 1; idx < degree; idx++) {
                                ff = x * ff + pt[idx];
                                df = x * df + ff;
                            } // End for i
                            ff = x * ff + pt[degree];
                            dx = ff / df;
                            x -= dx;
                        } while (Math.abs(dx / x) > 0.005); // End do-while loop

                        bnd = x;

                        // Compute the derivative as the initial K polynomial and do 5 steps with no shift
                        for (let idx = 1; idx < degree; idx++) {
                            K[idx] = ((degree - idx) * poly[idx]) / degree;
                        }
                        K[0] = poly[0];
                        aa = poly[degree];
                        bb = poly[NM1];
                        zerok = K[NM1] === 0 ? 1 : 0;

                        for (jj = 0; jj < 5; jj++) {
                            cc = K[NM1];
                            if (zerok) {
                                // Use unscaled form of recurrence
                                for (let idx = 0; idx < NM1; idx++) {
                                    j = NM1 - idx;
                                    K[j] = K[j - 1];
                                } // End for i
                                K[0] = 0;
                                zerok = K[NM1] === 0 ? 1 : 0;
                            } else {
                                // Used scaled form of recurrence if value of K at 0 is nonzero
                                t = -aa / cc;
                                for (let idx = 0; idx < NM1; idx++) {
                                    j = NM1 - idx;
                                    K[j] = t * K[j - 1] + poly[j];
                                } // End for i
                                K[0] = poly[0];
                                zerok = Math.abs(K[NM1]) <= Math.abs(bb) * DBL_EPSILON * 10.0 ? 1 : 0;
                            }
                        }

                        // Save K for restarts with new shifts
                        for (let idx = 0; idx < degree; idx++) {
                            temp[idx] = K[idx];
                        }

                        // Loop to select the quadratic corresponding to each new shift
                        for (jj = 1; jj <= 20; jj++) {
                            // Quadratic corresponds to a double shift to a non-real point and its
                            // complex conjugate. The point has modulus BND and amplitude rotated
                            // by 94 degrees from the previous shift.

                            xxx = -(sinr * yy) + cosr * xx;
                            yy = sinr * xx + cosr * yy;
                            xx = xxx;
                            sr = bnd * xx;
                            u = -(2.0 * sr);

                            // Second stage calculation, fixed quadratic
                            fxshfrAk1(DBL_EPSILON, MDP1, 20 * jj, sr, bnd, K, degree, poly, NN, qp, u, fxshfrPar);

                            if (fxshfrPar.NZ === 0) {
                                // If the iteration is unsuccessful, another quadratic is chosen after restoring K
                                for (let idx = 0; idx < degree; idx++) {
                                    K[idx] = temp[idx];
                                }
                            } else {
                                // The second stage jumps directly to one of the third stage iterations and
                                // returns here if successful. Deflate the polynomial, store the zero or
                                // zeros, and return to the main algorithm.
                                j = degPar.Degree - degree;
                                zeroReal[j] = fxshfrPar.szr;
                                zeroImag[j] = fxshfrPar.szi;
                                NN -= fxshfrPar.NZ;
                                degree = NN - 1;
                                for (let idx = 0; idx < NN; idx++) {
                                    poly[idx] = qp[idx];
                                }
                                if (fxshfrPar.NZ === 1) {
                                    // Single zero found, no additional zeros to store
                                } else {
                                    zeroReal[j + 1] = fxshfrPar.lzr;
                                    zeroImag[j + 1] = fxshfrPar.lzi;
                                }
                                break;
                            }
                        }
                        // Return with failure if no convergence with 20 shifts
                        if (jj > 20) {
                            degPar.Degree -= degree;
                            break;
                        }
                    }
                    // >>>>> End Main Loop <<<<<
                }
                // --> End Jenkins-Traub
                rpSolve(degreePar, p, zeror, zeroi);

                const l = zeroi.length;
                /** @type {(string | number)[]} */
                const results = [];
                // Format the output
                for (i = 0; i < l; i++) {
                    // We round the imaginary part to avoid having something crazy like 5.67e-16.
                    const img = round(zeroi[i], decp + 8);
                    let real = round(Number(zeror[i]), decp + 8);
                    // Did the rounding pay off? If the rounding did nothing more than chop off a few digits then no.
                    // If the rounding results in a a number at least 3 digits shorter we'll keep it else we'll keep
                    // the original otherwise the rounding was worth it.
                    real = decp - String(real).length > 2 ? real : Number(zeror[i]);
                    const sign = Number(img) < 0 ? '-' : '';

                    // Remove the zeroes
                    /** @type {string | number} */
                    let realStr = real;
                    /** @type {string | number} */
                    let imgStr = img;
                    if (real === 0) {
                        realStr = '';
                    }
                    if (img === 0) {
                        imgStr = '';
                    }

                    // Remove 1 as the multiplier and discard imaginary part if there isn't one.
                    if (Math.abs(Number(img)) === 1) {
                        imgStr = `${sign}i`;
                    } else if (img) {
                        imgStr = `${img}*i`;
                    } else {
                        imgStr = '';
                    }

                    const num = realStr && imgStr ? `${realStr}+${imgStr}` : String(realStr) + String(imgStr);
                    results[i] = num.replace(/\+-/gu, '-');
                }
                return results;
            }
        },
        roots(symbol) {
            if (symbol.isConstant(true, true)) {
                return core.Utils.nroots(symbol);
            }
            const roots = __.proots(symbol).map(x => _.parse(x));
            return core.Vector.fromArray(roots);
        },
        /**
         * Find root using Newton-Raphson method.
         *
         * @param {NerdamerSymbolType | ((x: number) => number)} f - Function or symbol
         * @param {number} guess - Initial guess
         * @param {((x: number) => number) | undefined} [dx] - Optional derivative
         * @returns {number | null}
         */
        froot(f, guess, dx) {
            /**
             * @param {number | null} xn
             * @returns {number | null}
             */
            const newtonraph = function (xn) {
                const mesh = 1e-12;
                // If the derivative was already provided then don't recalculate.
                const df = dx
                    ? dx
                    : core.Build.build(core.Calculus.diff(/** @type {NerdamerSymbolType} */ (f).clone()));
                // If the function was passed in as a function then don't recalculate.
                const fn = f instanceof Function ? f : core.Build.build(f);
                const max = 10000;
                let done = false;
                let safety = 0;
                while (!done) {
                    const x =
                        /** @type {number} */ (xn) - fn(/** @type {number} */ (xn)) / df(/** @type {number} */ (xn));
                    // Absolute values for both x & xn ensures that we indeed have the radius
                    const r = Math.abs(x) - Math.abs(/** @type {number} */ (xn));
                    const delta = Math.abs(r);
                    xn = x;

                    if (delta < mesh) {
                        done = true;
                    } else if (safety > max) {
                        xn = null;
                        done = true;
                    }

                    safety++;
                }
                return xn;
            };
            return newtonraph(Number(guess));
        },
        /**
         * Solve quadratic equation.
         *
         * @param {NerdamerSymbolType | string} a
         * @param {NerdamerSymbolType | string} b
         * @param {NerdamerSymbolType | string} c
         * @returns {NerdamerSymbolType[]}
         */
        quad(a, b, c) {
            /**
             * @param {NerdamerSymbolType | string} qa
             * @param {NerdamerSymbolType | string} qb
             * @param {NerdamerSymbolType | string} qc
             * @param {number} sign
             * @returns {NerdamerSymbolType}
             */
            const q = function (qa, qb, qc, sign) {
                return /** @type {NerdamerSymbolType} */ (
                    _.parse(`-(${qb}+${sign}*sqrt((${qb})^2-4*(${qa})*(${qc})))/(2*${qa})`)
                );
            };
            return [q(a, b, c, 1), q(a, b, c, -1)];
        },
        /**
         * Returns sum and product given roots.
         *
         * @param {NerdamerSymbolType | string} a
         * @param {NerdamerSymbolType | string} b
         * @returns {NerdamerSymbolType[]}
         */
        sumProd(a, b) {
            return __.quad(String(-b), String(a), '-1').map(x => x.invert());
        },
        coeffs(symbol, wrt, coeffs) {
            symbol = /** @type {NerdamerSymbolType} */ (_.expand(symbol));
            coeffs ||= [new NerdamerSymbol(0)];
            // We cannot get coeffs for group EX
            let vars = variables(symbol);

            // If wrt is not provided and there's only one variable, use it
            if (wrt === undefined && vars.length === 1) {
                wrt = vars[0];
            }
            wrt = String(wrt);

            if (symbol.group === EX && symbol.contains(wrt, true)) {
                _.error(`Unable to get coefficients using expression ${symbol.toString()}`);
            }
            vars = variables(symbol);

            // Check if symbol contains irrational constants that would be lost by Polynomial
            // These include pi, e, and sqrt (which are treated as constants but aren't simple numbers)
            const hasIrrationalConstants =
                symbol.contains('pi') || symbol.contains('e') || symbol.containsFunction('sqrt');

            if (vars.length === 1 && vars[0] === wrt && !symbol.isImaginary() && !hasIrrationalConstants) {
                const a = new Polynomial(symbol).coeffs.map(x => new NerdamerSymbol(x));

                for (let i = 0, l = a.length; i < l; i++) {
                    let coeff = a[i];
                    const e = coeffs[i];
                    if (e) {
                        coeff = /** @type {NerdamerSymbolType} */ (_.add(e, coeff));
                    }
                    coeffs[i] = coeff; // Transfer it all over
                }
            } else if (
                vars.length === 1 &&
                vars[0] === wrt &&
                !symbol.isImaginary() &&
                hasIrrationalConstants &&
                symbol.group === CP
            ) {
                // Use getCoeffs which properly preserves symbolic constants
                // Only for CP (sum) groups - CB (product) groups are handled in the else branch
                const a = core.Utils.getCoeffs(symbol, wrt);

                for (let i = 0, l = a.length; i < l; i++) {
                    let coeff = /** @type {NerdamerSymbolType} */ (a[i]);
                    const e = coeffs[i];
                    if (e) {
                        coeff = /** @type {NerdamerSymbolType} */ (_.add(e, coeff));
                    }
                    coeffs[i] = coeff;
                }
            } else {
                if (!wrt) {
                    _.error('Polynomial contains more than one variable. Please specify which variable is to be used!');
                }
                // If the variable isn't part of this polynomial then we're looking at x^0

                if (vars.indexOf(wrt) === -1) {
                    coeffs[0] = /** @type {NerdamerSymbolType} */ (_.add(symbol, coeffs[0]));
                } else {
                    coeffs ||= [new NerdamerSymbol(0)];
                    let coeff;
                    if (symbol.group === CB) {
                        const s = symbol.symbols[wrt];
                        if (!s) {
                            _.error('Expression is not a polynomial!');
                        }
                        const p = Number(s.power);
                        coeff = /** @type {NerdamerSymbolType} */ (_.divide(symbol.clone(), s.clone()));
                        if (/** @type {NerdamerSymbolType} */ (coeff).contains(wrt, true) || p < 0 || !isInt(p)) {
                            _.error('Expression is not a polynomial!');
                        }
                        const e = coeffs[p];
                        if (e) {
                            coeff = /** @type {NerdamerSymbolType} */ (_.add(e, coeff));
                        }
                        coeffs[p] = coeff;
                    } else if (symbol.group === CP) {
                        symbol.each(x => {
                            __.coeffs(x.clone(), wrt, coeffs);
                        }, true);
                    }
                }
            }
            // Fill holes
            for (let i = 0, l = coeffs.length; i < l; i++) {
                if (typeof coeffs[i] === 'undefined') {
                    coeffs[i] = new NerdamerSymbol(0);
                }
            }

            return coeffs;
        },
        /**
         * Get's all the powers of a particular polynomial including the denominators. The denominators powers are
         * returned as negative. All remaining polynomials are returned as zero order polynomials. for example
         * polyPowers(x^2+1/x+y+t) will return [ '-1', 0, '2' ]
         *
         * @param {NerdamerSymbolType} e
         * @param {string} forVariable
         * @param {Array} powers
         * @returns {Array} An array of the powers
         */
        // assumes you've already verified that it's a polynomial
        polyPowers(e, forVariable, powers) {
            powers ||= [];
            const g = e.group;
            if (g === PL && forVariable === e.value) {
                powers = powers.concat(keys(e.symbols));
            } else if (g === CP) {
                for (const s in e.symbols) {
                    if (!Object.hasOwn(e.symbols, s)) {
                        continue;
                    }
                    const symbol = e.symbols[s];
                    const symGroup = symbol.group;
                    const v = symbol.value;
                    if (symGroup === S && forVariable === v) {
                        powers.push(symbol.power);
                    } else if (symGroup === PL || symGroup === CP) {
                        powers = __.polyPowers(symbol, forVariable, powers);
                    } else if (symGroup === CB && symbol.contains(forVariable)) {
                        const t = symbol.symbols[forVariable];
                        if (t) {
                            powers.push(t.power);
                        }
                    } else if (symGroup === N || forVariable !== v) {
                        powers.push(0);
                    }
                }
            } else if (g === CB && e.contains(forVariable)) {
                const decomp = /** @type {DecomposeResultType} */ (core.Utils.decompose_fn(e, forVariable, true));
                powers.push(decomp.x.power);
            }
            return core.Utils.arrayUnique(powers).sort();
        },
        // The factor object
        Factor: {
            // Splits the symbol in symbol and constant
            split(symbol) {
                let c = new NerdamerSymbol(1); // The constants part
                let s = new NerdamerSymbol(1); // The symbolic part
                __.Factor.factorInner(symbol, new Factors()).each(x => {
                    const t = /** @type {NerdamerSymbolType} */ (_.parse(x));
                    if (x.isConstant(true)) {
                        c = /** @type {NerdamerSymbolType} */ (_.multiply(c, t));
                    } else {
                        s = /** @type {NerdamerSymbolType} */ (_.multiply(s, t));
                    }
                });
                return [c, s];
            },
            mix(o, includeNegatives) {
                const factors = keys(o);
                const l = factors.length;
                const m = []; // Create a row which we'r going to be mixing
                for (let i = 0; i < l; i++) {
                    const factor = Number(factors[i]);
                    const p = o[factors[i]];
                    const ll = m.length;
                    for (let j = 0; j < ll; j++) {
                        const t = m[j] * factor;
                        m.push(t);
                        if (includeNegatives) {
                            m.push(-t);
                        }
                    }

                    for (let j = 1; j <= p; j++) {
                        m.push(factor ** j);
                    }
                }
                return m;
            },
            // TODO: this method is to replace common factoring
            common(symbol, factors) {
                try {
                    if (symbol.group === CP) {
                        // This may have the unfortunate side effect of expanding and factoring again
                        // to only end up with the same result.
                        // TODO: try to avoid this
                        // collect the symbols and sort to have the longest first. Thinking is that the longest terms
                        // has to contain the variable in order for it to be factorable
                        const expanded = /** @type {NerdamerSymbolType} */ (
                            _.expand(symbol.clone(), { expand_denominator: true })
                        );
                        /** @type {(sym: unknown) => number} */
                        const getLength = sym => /** @type {{ length?: number }} */ (sym).length || 1;
                        const symbols = /** @type {NerdamerSymbolType[]} */ (
                            expanded.collectSymbols(null, null, (a, b) => getLength(b) - getLength(a))
                        );

                        /** @type {Record<string, [number, NerdamerSymbolType[]]>} */
                        const map = {}; // Create a map of common factors
                        /** @type {FracType[]} */
                        const coeffs = [];
                        for (let i = 0; i < symbols.length; i++) {
                            const sym = symbols[i];
                            coeffs.push(sym.multiplier.clone());
                            sym.each(x => {
                                const p = Number(x.power);
                                // This check exits since we have a symbolic power.
                                // For the future... think about removing this check and modify for symbolic powers
                                if (isNaN(p)) {
                                    throw new Error('exiting');
                                }
                                // Loop through the symbols and lump together common terms
                                if (x.value in map) {
                                    if (p < map[x.value][0]) {
                                        map[x.value][0] = p;
                                    }
                                    map[x.value][1].push(x);
                                } else {
                                    map[x.value] = [p, [x]];
                                }
                            });
                        }
                        // The factor
                        let factor = new NerdamerSymbol(1);
                        for (const x in map) {
                            // If this factor is found in all terms since the length of
                            // matching variable terms matches the number of original terms
                            if (map[x][1].length === symbols.length) {
                                // Generate a symbol and multiply into the factor
                                factor = /** @type {NerdamerSymbolType} */ (
                                    _.multiply(
                                        factor,
                                        /** @type {NerdamerSymbolType} */ (
                                            _.pow(new NerdamerSymbol(x), new NerdamerSymbol(map[x][0]))
                                        )
                                    )
                                );
                            }
                        }
                        // Get coefficient factor
                        const c = core.Math2.QGCD.apply(null, coeffs);

                        if (!c.equals(1)) {
                            factors.add(new NerdamerSymbol(c));
                            for (let i = 0; i < symbols.length; i++) {
                                symbols[i].multiplier = symbols[i].multiplier.divide(c);
                            }
                        }

                        // If we actuall found any factors
                        if (!factor.equals(1)) {
                            factors.add(factor);
                            symbol = new NerdamerSymbol(0);
                            for (let i = 0; i < symbols.length; i++) {
                                symbol = /** @type {NerdamerSymbolType} */ (
                                    _.add(symbol, _.divide(symbols[i], factor.clone()))
                                );
                            }
                        }
                    }
                } catch (e) {
                    if (e.message === 'timeout') {
                        throw e;
                    }
                }

                return symbol;
            },
            zeroes(symbol, factors) {
                const exit = function () {
                    throw new core.exceptions.ValueLimitExceededError('Exiting');
                };
                try {
                    let term;
                    let sum;
                    let p;
                    symbol = /** @type {NerdamerSymbolType} */ (_.expand(symbol.clone()));
                    const e = symbol.toString();
                    const vars = variables(symbol);

                    sum = new NerdamerSymbol(0);

                    const terms = [];
                    /** @type {FracType[]} */
                    const powers = [];

                    // Start setting each variable to zero
                    for (let i = 0, l = vars.length; i < vars.length; i++) {
                        /** @type {Record<string, ExpressionParam>} */
                        const subs = {};
                        // We want to create a subs object with all but the current variable set to zero
                        for (let j = 0; j < l; j++) {
                            if (i !== j) // Make sure we're not looking at the same variable
                            {
                                subs[vars[j]] = 0;
                            }
                        }
                        term = /** @type {NerdamerSymbolType} */ (_.parse(e, subs));
                        const tp = term.power;
                        // The temporary power has to be an integer as well
                        if (!isInt(tp)) {
                            exit();
                        }
                        terms.push(term);
                        powers.push(/** @type {FracType} */ (term.power));
                    }

                    // Get the gcd. This will be the p in (a^n+b^m)^p
                    // if the gcd equals 1 meaning n = m then we need a tie breakder
                    if (core.Utils.allSame(powers)) {
                        // Get p given x number of terms
                        const nTerms = symbol.length;
                        // The number of zeroes determines
                        const nZeroes = terms.length;
                        const den = Math.round((Math.sqrt(8 * nTerms - 1) - 3) / 2);
                        if (nZeroes === 2) {
                            p = new Frac(Number(powers[0]) / (nTerms - 1));
                        } else if (nZeroes === 3 && den !== 0) {
                            p = new Frac(Number(powers[0]) / den);
                        } else {
                            // P is just the gcd of the powers
                            p = core.Math2.QGCD.apply(null, /** @type {FracType[]} */ (powers));
                        }
                        /*
                         //get the lowest possible power
                         //e.g. given b^4+2*a^2*b^2+a^4, the power we're looking for would be 2
                         symbol.each(function(x) {
                         if(x.group === CB)
                         x.each(function(y) {
                         if(!p || y.power.lessThan(p))
                         //p = Number(y.power);
                         p = y.power;
                         });
                         else if(!p || x.power.lessThan(p))
                         //p = Number(x.power);
                         p = x.power;
                         });
                         */
                    } else {
                        // P is just the gcd of the powers
                        p = core.Math2.QGCD.apply(null, powers);
                    }

                    // If we don't have an integer then exit
                    if (!isInt(p)) {
                        return symbol; // Nothing to do
                        // exit();
                    }

                    // Build the factor
                    for (let i = 0; i < terms.length; i++) {
                        const t = terms[i];
                        const nFrac = /** @type {FracType} */ (t.power).clone().divide(/** @type {FracType} */ (p));
                        const n = Number(nFrac);
                        // Don't take squareroots of negatives
                        if ((Number(t.multiplier.num) < 0 || Number(t.multiplier.den) < 0) && n % 2 === 0) {
                            return symbol;
                        }
                        t.multiplier = new Frac(Number(t.multiplier) ** (1 / n));
                        t.power = /** @type {FracType} */ (p).clone();
                        sum = /** @type {NerdamerSymbolType} */ (_.add(sum, t));
                    }

                    // By now we have the factor of zeroes. We'll know if we got it right because
                    // we'll get a remainder of zero each time we divide by it
                    if (/** @type {NerdamerSymbolType} */ (sum).group !== CP) {
                        return symbol;
                    } // Nothing to do

                    while (true) {
                        const d = __.div(symbol.clone(), sum.clone());
                        if (/** @type {NerdamerSymbolType} */ (d[1]).equals(0)) {
                            symbol = /** @type {NerdamerSymbolType} */ (d[0]);
                            factors.add(sum.clone());
                            if (symbol.equals(1)) // We've reached 1 so done.
                            {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                } catch (e) {
                    if (e.message === 'timeout') {
                        throw e;
                    }
                }
                return symbol;
            },
            factor(symbol, factors) {
                core.Utils.checkTimeout();
                const originalFactors = factors ? { ...factors.factors } : null;
                const originalLength = factors ? factors.length : 0;
                try {
                    let retval = __.Factor.factorInner(symbol, factors);
                    retval = retval.pushMinus();
                    return retval;
                } catch (error) {
                    if (error.message === 'timeout') {
                        throw error;
                    }

                    if (factors && originalFactors) {
                        factors.factors = originalFactors;
                        factors.length = originalLength;
                    }
                    return symbol;
                }
            },
            factorInner(symbol, factors) {
                core.Utils.checkTimeout();
                // Don't try to factor constants,
                // do it with Math2.factor
                if (symbol.isConstant()) {
                    if (symbol.isInteger()) {
                        return core.Math2.factor(Number(symbol.multiplier));
                    }
                    // Return symbol;
                }

                const _symbol = /** @type {NerdamerSymbolType} */ (_.parse(symbol));

                // Functions may have been evaluated in parse()
                // STILL don't try to factor constants
                // do it with Math2.factor
                if (_symbol.isConstant()) {
                    if (_symbol.isInteger()) {
                        return core.Math2.factor(Number(_symbol.multiplier));
                    }
                    return symbol;
                }

                // Shortcut 0 and 1
                if (_symbol.equals(0) || _symbol.equals(1)) {
                    return _symbol;
                }

                let retval = __.Factor._factor(_symbol, factors);
                if (retval.equals(symbol)) {
                    return retval;
                }

                // Shortcut 0 and 1 AGAIN after factor (which does eval)
                if (retval.equals(0) || retval.equals(1)) {
                    return retval;
                }

                if (retval.group === CB) {
                    let t = new NerdamerSymbol(1);
                    const p = _.parse(retval.power);
                    // Store the multiplier and strip it
                    let m = _.parse(retval.multiplier);

                    retval.toUnitMultiplier();

                    /*
                     * NOTE: for sign issues with factor START DEBUGGING HERE
                     */
                    // move the sign to t
                    if (retval.multiplier.lessThan(0)) {
                        t.negate();
                        retval.negate();
                    }

                    retval.each(x => {
                        // Related to #566. Since the symbol's group may not have been properly
                        // updated, it's easier to just parse the symbol and have the parser
                        // do the update for us.

                        const factored = /** @type {NerdamerSymbolType} */ (_.parse(__.Factor._factor(x)));
                        m = /** @type {NerdamerSymbolType} */ (
                            _.multiply(m, NerdamerSymbol.create(factored.multiplier.toString()))
                        );
                        factored.toUnitMultiplier();

                        if (factored.group === CB) {
                            let _t = new NerdamerSymbol(1);
                            factored.each(y => {
                                const _factored = /** @type {NerdamerSymbolType} */ (_.parse(__.Factor._factor(y)));
                                if (_factored.group === CB) {
                                    m = /** @type {NerdamerSymbolType} */ (
                                        _.multiply(m, NerdamerSymbol.create(_factored.multiplier.toString()))
                                    );
                                    _factored.toUnitMultiplier();
                                }
                                _t = /** @type {NerdamerSymbolType} */ (_.multiply(_t, _factored));
                            });
                            _t = /** @type {NerdamerSymbolType} */ (
                                _.pow(_t, new NerdamerSymbol(factored.power.toString()))
                            );
                            t = /** @type {NerdamerSymbolType} */ (_.multiply(t, _t));
                        } else {
                            t = /** @type {NerdamerSymbolType} */ (_.multiply(t, factored));
                        }
                    });

                    // Put back the multiplier and power
                    const pow = /** @type {NerdamerSymbolType} */ (_.pow(t, p));
                    retval = /** @type {NerdamerSymbolType} */ (_.multiply(m, pow));
                }
                return retval;
            },
            quadFactor(symbol, factors) {
                if (symbol.isPoly() && __.degree(symbol).equals(2)) {
                    // We've  already checked that we're dealing with a polynomial
                    const v = core.Utils.variables(symbol)[0]; // Get the variable
                    const coeffs = __.coeffs(symbol, v);
                    // Factor the lead coefficient
                    if (coeffs.length < 3) {
                        return symbol;
                    }
                    const cf = __.Factor._factor(coeffs[2].clone());
                    // Check if we have factors
                    if (cf.group === CB) {
                        const symbols = /** @type {NerdamerSymbolType[]} */ (cf.collectSymbols());
                        // If the factors are greater than 2 we're done so exit
                        if (symbols.length > 2) {
                            return symbol;
                        }
                        // If we have two factors then attempt to factor the polynomial
                        // let the factors be f1 and f1
                        // let the factors be (ax+b)(cx+d)
                        // let the coefficients be c1x^2+c2x+c3
                        // then a(x1)+c(x2)=c2 and x1*x2=c3
                        // we can solve for x1 and x2
                        const c = /** @type {NerdamerSymbolType} */ (
                            _.multiply(_.parse(coeffs[0]), _.parse(symbols[0]))
                        );
                        const b = /** @type {NerdamerSymbolType} */ (_.parse(coeffs[1])).negate();
                        const a = /** @type {NerdamerSymbolType} */ (_.parse(symbols[1]));
                        // Solve the system
                        const root = __.quad(a, b, c).filter(x => core.Utils.isInt(x));
                        // If we have one root then find the other one by dividing the constant
                        if (root.length === 1) {
                            const root1 = root[0];
                            const root2 = _.divide(coeffs[0], /** @type {NerdamerSymbolType} */ (_.parse(root1)));
                            if (core.Utils.isInt(root2)) {
                                // We found them both
                                factors.add(
                                    /** @type {NerdamerSymbolType} */ (
                                        _.parse(format('({0})*({1})+({2})', String(symbols[1]), v, String(root2)))
                                    )
                                );
                                factors.add(
                                    /** @type {NerdamerSymbolType} */ (
                                        _.parse(format('({0})*({1})+({2})', String(symbols[0]), v, root1))
                                    )
                                );
                                symbol = new NerdamerSymbol(1);
                            }
                        }
                    }
                    // // sanitization: eliminate "-(-x)"
                    // for (let xk in symbol.symbols) {
                    //     let x = symbol.symbols[xk];
                    //     if ((x.group === CB || x.group === CP || x.group === PL) &&
                    //         x.multiplier.equals(-1)) {
                    //         console.log("replacing "+x)
                    //         symbol[xk] = _.parse(x);
                    //         console.log("with "+symbol[xk])
                    //     }
                    // }
                }
                return symbol;
            },
            cubeFactor(symbol, factors) {
                if (symbol.isComposite()) {
                    const symbols = /** @type {NerdamerSymbolType[]} */ (symbol.collectSymbols());
                    // The symbol should be in the form of a^3+-b^3. The length
                    // should therefore only be two. If it's any different from this
                    // then we're done
                    if (symbols.length === 2) {
                        // Store the signs and then strip them from the symbols
                        let signA = symbols[0].sign();
                        let a = symbols[0].clone().abs();
                        let signB = symbols[1].sign();
                        let b = symbols[1].clone().abs();
                        // Check if they're cube
                        if (a.isCube() && b.isCube()) {
                            // Keep the negative sign on the right, meaning b is always negative.
                            if (signA < signB) {
                                // Swap the signs and then the values
                                [signA, signB] = [signB, signA];
                                [a, b] = [b, a];
                            }

                            // Get teh roots
                            const mRootA = _.parse(a.getNth(3));
                            const mRootB = _.parse(b.getNth(3));

                            // Remove the cube for both
                            const x = _.multiply(_.expand(_.pow(a.clone().toUnitMultiplier(), _.parse('1/3'))), mRootA);
                            const y = _.multiply(_.expand(_.pow(b.clone().toUnitMultiplier(), _.parse('1/3'))), mRootB);

                            if (signA === 1 && signB === -1) {
                                // Apply difference of cubes rule
                                factors.add(_.parse(format('(({0})-({1}))', String(x), String(y))));
                                factors.add(_.parse(format('(({0})^2+({0})*({1})+({1})^2)', String(x), String(y))));
                                symbol = new NerdamerSymbol(1);
                            } else if (signA === 1 && signB === 1) {
                                // Apply sum of cubes rule
                                factors.add(_.parse(format('(({0})+({1}))', String(x), String(y))));
                                factors.add(_.parse(format('(({0})^2-({0})*({1})+({1})^2)', String(x), String(y))));
                                symbol = new NerdamerSymbol(1);
                            }
                        }
                    }
                }

                return symbol;
            },
            /**
             * Internal factorization implementation
             *
             * @param {NerdamerSymbolType} symbol
             * @param {FactorsLike} [factors]
             * @returns {NerdamerSymbolType}
             */
            _factor(symbol, factors) {
                core.Utils.checkTimeout();
                const _g = symbol.group;
                // Some items cannot be factored any further so return those right away
                if (symbol.group === FN) {
                    const arg = symbol.args[0];
                    if (arg.group === S && arg.isSimple()) {
                        return symbol;
                    }
                } else if (symbol.group === S && symbol.isSimple()) {
                    return symbol;
                }

                // Expand the symbol to get it in a predictable form. If this step
                // is skipped some factors are missed.
                // if(symbol.group === CP && !(even(symbol.power) && symbol.multiplier.lessThan(0))) {
                if (symbol.group === CP) {
                    symbol.distributeMultiplier(true);
                    let t = new NerdamerSymbol(0);
                    symbol.each(x => {
                        if ((x.group === CP && x.power.greaterThan(1)) || x.group === CB) {
                            x = /** @type {NerdamerSymbolType} */ (_.expand(x));
                        }
                        t = /** @type {NerdamerSymbolType} */ (_.add(t, x));
                    });
                    t.power = symbol.power;

                    symbol = t;
                }

                if (symbol.group === FN && symbol.fname !== 'sqrt') {
                    symbol = core.Utils.evaluate(symbol);
                }

                // Make a copy of the symbol to return if something goes wrong
                const untouched = symbol.clone();
                try {
                    if (symbol.group === CB) {
                        const _p = _.parse(symbol.power);

                        // Grab the denominator and strip the multiplier and power. Store them in an array
                        const denArray = __.Simplify.strip(symbol.getDenom());
                        const numArray = __.Simplify.strip(symbol.getNum());

                        const den = denArray.pop();
                        const num = numArray.pop();

                        // If the numerator equals the symbol then we've hit the simplest form and then we're done
                        if (num.equals(symbol)) {
                            return symbol;
                        }
                        const nfact = __.Factor.factorInner(num);
                        const dfact = __.Factor.factorInner(den);

                        const n = __.Simplify.unstrip(
                            /** @type {[NerdamerSymbolType, NerdamerSymbolType]} */ (/** @type {unknown} */ (numArray)),
                            nfact
                        );
                        const d = __.Simplify.unstrip(
                            /** @type {[NerdamerSymbolType, NerdamerSymbolType]} */ (/** @type {unknown} */ (denArray)),
                            dfact
                        );

                        const retval = /** @type {NerdamerSymbolType} */ (_.divide(n, d));

                        return retval;
                    }
                    if (symbol.group === S) {
                        return symbol; // Absolutely nothing to do
                    }

                    if (symbol.isConstant()) {
                        if (symbol.equals(1) || symbol.equals(0) || !symbol.isInteger()) {
                            return symbol.clone();
                        }
                        const ret = core.Math2.factor(Number(symbol.multiplier));
                        return ret;
                    }

                    const p = symbol.power.clone();

                    if (isInt(p) && !(p.lessThan(0) && symbol.group === FN)) {
                        const sign = p.sign();
                        symbol.toLinear();
                        factors ||= new Factors();
                        /** @type {Record<string, string>} */
                        const map = {};
                        symbol = /** @type {NerdamerSymbolType} */ (_.parse(core.Utils.subFunctions(symbol, map)));
                        if (keys(map).length > 0) {
                            // It might have functions
                            factors.preAdd = function preAdd(factor) {
                                const ret = _.parse(factor, core.Utils.getFunctionsSubs(map));
                                return /** @type {NerdamerSymbolType} */ (ret);
                            };
                        }

                        // Strip the power
                        if (!symbol.isLinear()) {
                            factors.pFactor = symbol.power.toString();
                            symbol.toLinear();
                        }

                        const vars = variables(symbol);
                        // Bypass for imaginary. TODO: find a better solution
                        if (symbol.isImaginary()) {
                            vars.push(core.Settings.IMAGINARY);
                        }
                        const multiVar = vars.length > 1;

                        // Minor optimization. Seems to cut factor time by half in some cases.
                        if (multiVar) {
                            let allS = true;
                            let allUnit = true;
                            symbol.each(x => {
                                if (x.group !== S) {
                                    allS = false;
                                }
                                if (!x.multiplier.equals(1)) {
                                    allUnit = false;
                                }
                            });

                            if (allS && allUnit) {
                                return /** @type {NerdamerSymbolType} */ (
                                    _.pow(_.parse(symbol, core.Utils.getFunctionsSubs(map)), _.parse(p))
                                );
                            }
                        }

                        // Factor the coefficients
                        const coeffFactors = new Factors();

                        symbol = __.Factor.coeffFactor(symbol, coeffFactors);

                        coeffFactors.each(x => {
                            // If the factor was negative but was within a square then it becomes positive
                            if (even(Number(p)) && x.lessThan(0)) {
                                x.negate();
                            }

                            if (sign < 0) {
                                x.invert();
                            }
                            factors.add(x);
                        });

                        // Factor the power
                        const powerFactors = new Factors();
                        symbol = __.Factor.powerFactor(symbol, powerFactors);
                        powerFactors.each(x => {
                            if (sign < 0) {
                                x.invert();
                            }
                            factors.add(x);
                        });

                        if (multiVar) {
                            // Try sum and difference of cubes
                            symbol = __.Factor.cubeFactor(symbol, factors);

                            symbol = __.Factor.mfactor(symbol, factors);

                            // Put back the sign of power
                            factors.each(x => {
                                if (sign < 0) {
                                    x.power.negate();
                                }
                            });
                        } else {
                            // Pass in vars[0] for safety
                            const v = vars[0];

                            symbol = __.Factor.squareFree(symbol, factors, v);

                            const tFactors = new Factors();

                            symbol = __.Factor.trialAndError(symbol, tFactors, v);

                            // Generate a symbol based off the last factors
                            const tfSymbol = tFactors.toSymbol();
                            // If nothing was factored then return the factors
                            if (tfSymbol.equals(untouched)) {
                                return tfSymbol;
                            }

                            for (const x in tFactors.factors) {
                                if (!Object.hasOwn(tFactors.factors, x)) {
                                    continue;
                                }
                                // Store the current factor in tFactor
                                const tFactor = tFactors.factors[x];
                                factors.add(/** @type {NerdamerSymbolType} */ (_.pow(tFactor, _.parse(p))));
                            }
                            // If we still don't have a factor and it's quadratic then let's just do a quad factor
                            if (symbol.equals(untouched)) {
                                symbol = __.Factor.quadFactor(symbol, factors);
                            }
                        }

                        // Last minute clean up
                        symbol = /** @type {NerdamerSymbolType} */ (_.parse(symbol, core.Utils.getFunctionsSubs(map)));

                        const addPower = factors.length === 1;

                        factors.add(/** @type {NerdamerSymbolType} */ (_.pow(symbol, _.parse(p))));

                        let retval = factors.toSymbol();

                        // We may have only factored out the symbol itself so we end up with a factor of one
                        // where the power needs to be placed back
                        // e.g. factor((2*y+p)^2). Here we end up having a factor of 1 remaining and a p of 2.
                        if (addPower && symbol.equals(1) && retval.isLinear()) {
                            retval = /** @type {NerdamerSymbolType} */ (_.pow(retval, _.parse(p)));
                        }

                        return retval;
                    }

                    return symbol;
                } catch (e) {
                    if (e?.message === 'timeout') {
                        throw e;
                    }
                    // No need to stop the show because something went wrong :). Just return the unfactored.
                    return untouched;
                }
            },
            reduce(symbol, factors) {
                if (symbol.group === CP && symbol.length === 2) {
                    const symbols = /** @type {NerdamerSymbolType[]} */ (symbol.collectSymbols()).sort(
                        (a, b) => Number(b.multiplier) - Number(a.multiplier)
                    );
                    if (/** @type {FracType} */ (symbols[0].power).equals(/** @type {FracType} */ (symbols[1].power))) {
                        // X^n-a^n
                        const n = /** @type {NerdamerSymbolType} */ (_.parse(symbols[0].power));
                        const a = symbols[0].clone().toLinear();
                        const b = symbols[1].clone().toLinear();

                        // Apply rule: (a-b)*sum(a^(n-i)*b^(i-1),1,n)
                        factors.add(/** @type {NerdamerSymbolType} */ (_.add(a.clone(), b.clone())));
                        // Flip the sign
                        b.negate();
                        // Turn n into a number
                        const nn = Number(n);
                        // The remainder
                        let result = new NerdamerSymbol(0);
                        for (let i = 1; i <= nn; i++) {
                            const aa = /** @type {NerdamerSymbolType} */ (
                                _.pow(a.clone(), _.subtract(n.clone(), new NerdamerSymbol(i)))
                            );
                            const bb = /** @type {NerdamerSymbolType} */ (
                                _.pow(b.clone(), _.subtract(new NerdamerSymbol(i), new NerdamerSymbol(1)))
                            );
                            result = /** @type {NerdamerSymbolType} */ (
                                _.add(result, /** @type {NerdamerSymbolType} */ (_.multiply(aa, bb)))
                            );
                        }
                        return result;
                    }
                }
                return symbol;
            },
            /**
             * Makes NerdamerSymbol square free
             *
             * @param {NerdamerSymbolType} symbol
             * @param {Factors} factors
             * @param {string} [variable] The variable which is being factored
             * @returns {NerdamerSymbolType}
             */
            squareFree(symbol, factors, variable) {
                if (symbol.isConstant() || symbol.group === S) {
                    return symbol;
                }

                if (!symbol.isPoly()) {
                    return symbol;
                }

                const poly = new Polynomial(symbol, variable);
                const sqfr = poly.squareFree();
                const p = sqfr[2];
                // If we found a square then the p entry in the array will be non-unit
                if (p !== 1) {
                    // Make sure the remainder doesn't have factors
                    const t = sqfr[1].toSymbol();
                    t.power = /** @type {FracType} */ (t.power).multiply(new Frac(p));
                    // Send the factor to be fatored to be sure it's completely factored
                    factors.add(__.Factor.factorInner(t));

                    const retval = __.Factor.squareFree(sqfr[0].toSymbol(), factors);

                    return retval;
                }

                return symbol;
            },
            /**
             * Factors the powers such that the lowest power is a constant
             *
             * @param {NerdamerSymbolType} symbol
             * @param {Factors} factors
             * @returns {NerdamerSymbolType}
             */
            powerFactor(symbol, factors) {
                // Only PL need apply
                if (symbol.group !== PL || symbol.previousGroup === EX) {
                    return symbol;
                }
                const k = keys(symbol.symbols);
                // We expect only numeric powers so return all else
                if (!core.Utils.allNumeric(k)) {
                    return symbol;
                }

                const d = core.Utils.arrayMin(/** @type {number[]} */ (/** @type {unknown} */ (k)));
                let retval = new NerdamerSymbol(0);
                const q = /** @type {NerdamerSymbolType} */ (_.parse(`${symbol.value}^${d}`));
                symbol.each(x => {
                    x = /** @type {NerdamerSymbolType} */ (_.divide(x, q.clone()));
                    retval = /** @type {NerdamerSymbolType} */ (_.add(retval, x));
                });

                factors.add(q);
                return retval;
            },
            /**
             * Removes GCD from coefficients
             *
             * @param {NerdamerSymbolType} symbol
             * @param {Factors} factors
             * @returns {NerdamerSymbolType}
             */
            coeffFactor(symbol, factors) {
                if (symbol.isComposite()) {
                    const gcd = core.Math2.QGCD.apply(null, symbol.coeffs());

                    if (gcd.equals(1)) {
                        // TODO: This should probably go to the prototype
                        const power = function (sym) {
                            let p;
                            if (sym.group === CB) {
                                p = 0;
                                sym.each(x => {
                                    p += x.power;
                                });
                            } else {
                                p = Number(sym.power);
                            }
                            return p;
                        };
                        // Factor out negatives from the lead term
                        const terms = /** @type {NerdamerSymbolType[]} */ (
                            symbol.collectSymbols(null, null, null, true)
                        ).sort((a, b) => {
                            // Push constants to the back
                            if (a.isConstant(true)) {
                                return 1;
                            }
                            return Number(b.power) - Number(a.power);
                        });

                        const LT = terms[0];

                        // Check if the LT is indeed the greatest
                        if (power(LT) > power(terms[1]) || terms[1].isConstant(true)) {
                            if (LT.multiplier.lessThan(0)) {
                                // Although the symbol should always be linear at this point, remove the negative for squares
                                // to be safe.
                                factors.add(new NerdamerSymbol(-1));

                                symbol.each(x => {
                                    x.negate();
                                }, true);
                            }
                        }
                    } else {
                        symbol.each(x => {
                            if (x.isComposite()) {
                                x.each(y => {
                                    y.multiplier = y.multiplier.divide(gcd);
                                });
                            } else {
                                x.multiplier = x.multiplier.divide(gcd);
                            }
                        });
                        symbol.updateHash();
                    }

                    if (factors) {
                        factors.add(new NerdamerSymbol(gcd));
                    }
                }

                return symbol;
            },
            /**
             * The name says it all :)
             *
             * @param {NerdamerSymbolType} symbol
             * @param {Factors} factors
             * @param {string} variable
             * @returns {NerdamerSymbolType}
             */
            trialAndError(symbol, factors, variable) {
                const untouched = symbol.clone();
                try {
                    // At temp holder for the factors. If all goes well then
                    // they'll be moved to the actual factors.
                    const factorArray = [];

                    if (symbol.isConstant() || symbol.group === S || !symbol.isPoly()) {
                        return symbol;
                    }
                    let poly = new Polynomial(symbol, variable);
                    const cnst = poly.coeffs[0];
                    const cfactors = core.Math2.ifactor(Number(cnst));
                    const roots = __.proots(symbol);
                    for (let i = 0; i < roots.length; i++) {
                        let r = roots[i];
                        /** @type {number} */
                        let p = 1;
                        if (!isNaN(Number(r))) {
                            // If it's a number
                            for (const x in cfactors) {
                                if (!Object.hasOwn(cfactors, x)) {
                                    continue;
                                }
                                // Check it's raised to a power
                                const n = core.Utils.round(Math.log(Number(x)) / Math.log(Math.abs(Number(r))), 8);
                                if (isInt(n)) {
                                    r = x; // X must be the root since n gave us a whole
                                    p = Number(n);
                                    break;
                                }
                            }
                            const root = new Frac(Number(r));
                            const terms = [new Frac(Number(root.num)).negate()];
                            terms[p] = new Frac(Number(root.den));
                            // Convert to Frac. The den is coeff of LT and the num is coeff of constant
                            const div = Polynomial.fromArray(terms, poly.variable).fill();
                            const t = poly.divide(div);
                            if (t[1].equalsNumber(0)) {
                                // If it's zero we have a root and divide it out
                                poly = t[0];
                                // Factors.add(div.toSymbol());
                                factorArray.push(div.toSymbol());
                            }
                        }
                    }

                    if (!poly.equalsNumber(1)) {
                        poly = __.Factor.search(poly, factors);
                    }

                    // Move the factors over since all went well.
                    factorArray.forEach(x => {
                        factors.add(x);
                    });

                    return poly.toSymbol();
                } catch (e) {
                    if (e.message === 'timeout') {
                        throw e;
                    }
                    return untouched;
                }
            },
            search(poly, factors, base) {
                base ||= 10; // I like 10 because numbers exhibit similar behaviours at 10
                const v = poly.variable; // The polynmial variable name
                /**
                 * Attempt to remove a root by division given a number by first creating a polynomial fromt he given
                 * information
                 *
                 * @param {number} c1 - Coeffient for the constant
                 * @param {number} c2 - Coefficient for the LT
                 * @param {number} n - The number to be used to construct the polynomial
                 * @param {number} p - The power at which to create the polynomial
                 * @returns {null | [Polynomial, Polynomial]} - Returns polynomial array if successful otherwise null
                 */
                const check = function (c1, c2, n, p) {
                    const candidate = Polynomial.fit(c1, c2, n, base, p, v);
                    if (candidate && candidate.coeffs.length > 1) {
                        const t = poly.divide(candidate);
                        if (t[1].equalsNumber(0)) {
                            factors.add(candidate.toSymbol());
                            return [t[0], candidate];
                        }
                    }
                    return null;
                };
                const cnst = poly.coeffs[0];
                const cfactors = core.Math2.ifactor(Number(cnst));
                const lc = poly.lc();
                const ltfactors = core.Math2.ifactor(Number(lc));
                const subbed = poly.sub(base);
                const isubbed = core.Math2.ifactor(/** @type {number} */ (/** @type {unknown} */ (subbed)));
                const nfactors = __.Factor.mix(isubbed, /** @type {number} */ (/** @type {unknown} */ (subbed)) < 0);
                let cp = Math.ceil(poly.coeffs.length / 2);
                const lcIsNeg = lc.lessThan(0);
                const cnstIsNeg = cnst.lessThan(0);
                ltfactors['1'] = 1;
                cfactors['1'] = 1;
                while (cp--) {
                    for (const x in ltfactors) {
                        if (!Object.hasOwn(ltfactors, x)) {
                            continue;
                        }
                        for (const y in cfactors) {
                            if (!Object.hasOwn(cfactors, y)) {
                                continue;
                            }
                            for (let i = 0; i < nfactors.length; i++) {
                                let factorFound = check(Number(x), Number(y), nfactors[i], cp);
                                if (factorFound) {
                                    poly = factorFound[0];
                                    if (
                                        !core.Utils.isPrime(
                                            /** @type {number} */ (/** @type {unknown} */ (poly.sub(base)))
                                        )
                                    ) {
                                        poly = __.Factor.search(poly, factors);
                                    }
                                    return poly;
                                }
                                if (!factorFound) {
                                    if (lcIsNeg && cnstIsNeg) {
                                        factorFound = check(-Number(x), -Number(y), nfactors[i], cp);
                                    } else if (lcIsNeg) {
                                        factorFound = check(-Number(x), Number(y), nfactors[i], cp);
                                    } // Check a negative lc
                                    else if (cnstIsNeg) {
                                        factorFound = check(Number(x), -Number(y), nfactors[i], cp);
                                    } // Check a negative constant
                                }
                            }
                        }
                    }
                }
                return poly;
            },
            /**
             * Equivalent of square free factor for multivariate polynomials
             *
             * @param {NerdamerSymbolType} symbol
             * @param {Factors} factors
             * @returns {NerdamerSymbolType}
             */
            mSqfrFactor(symbol, factors) {
                if (symbol.group !== FN) {
                    const vars = variables(symbol).reverse();

                    // Loop through all the variable and remove the partial derivatives
                    for (let i = 0; i < vars.length; i++) {
                        let isFactor = false;
                        do {
                            if (vars[i] === symbol.value) {
                                // The derivative tells us nothing since this symbol is already the factor
                                factors.add(symbol);
                                symbol = new NerdamerSymbol(1);
                                continue;
                            }

                            const diff = core.Calculus.diff(symbol, vars[i]);

                            const d = __.Factor.coeffFactor(diff);

                            if (d.equals(0)) {
                                break;
                            }

                            // Sometimes nerdamer get too happy about factoring out 1 and -1
                            if (d.equals(1) || d.equals(-1)) {
                                break;
                            }

                            // Trial division to see if factors have whole numbers.
                            // This can be optimized by stopping as soon as canDivide is false
                            // this will also need utilize big number at some point
                            let canDivide = true;
                            if (d.isConstant() && symbol.isComposite()) {
                                // Check the coefficients

                                symbol.each(x => {
                                    if (Number(x.multiplier) % Number(d.multiplier) !== 0) {
                                        canDivide = false;
                                    }
                                }, true);
                            }

                            // If we can divide then do so
                            let div;
                            if (canDivide) {
                                const s = symbol.clone();
                                div = __.divWithCheck(symbol, d.clone());
                                isFactor = /** @type {NerdamerSymbolType} */ (div[1]).equals(0);

                                // Break infinite loop for factoring e^t*x-1
                                if (
                                    symbol.equals(/** @type {NerdamerSymbolType} */ (div[0])) &&
                                    /** @type {NerdamerSymbolType} */ (div[1]).equals(0)
                                ) {
                                    // Restore symbol, was mangled in __.div
                                    symbol = s;
                                    break;
                                }

                                if (/** @type {NerdamerSymbolType} */ (div[0]).isConstant()) {
                                    factors.add(/** @type {NerdamerSymbolType} */ (div[0]));
                                    break;
                                }
                            } else {
                                isFactor = false;
                            }

                            if (isFactor) {
                                factors.add(/** @type {NerdamerSymbolType} */ (div[0]));
                                symbol = d;
                            }
                        } while (isFactor);
                    }
                }

                return symbol;
            },
            // Difference of squares factorization
            sqdiff(symbol, factors) {
                if (symbol.isConstant('all')) {
                    // Nothing to do
                    return symbol;
                }

                try {
                    const removeSquare = function (x) {
                        return core.Utils.block(
                            'POSITIVE_MULTIPLIERS',
                            () => NerdamerSymbol.unwrapPARENS(math.sqrt(math.abs(x))),
                            true
                        );
                    };
                    const separated = core.Utils.separate(symbol.clone());
                    if (!separated) {
                        return symbol;
                    }

                    const objArray = [];

                    // Get the unique variables
                    for (const x in separated) {
                        if (x !== 'constants') {
                            objArray.push(separated[x]);
                        }
                    }
                    objArray.sort((a, b) => Number(b.power) - Number(a.power));

                    // If we have the same number of variables as unique variables then we can apply the difference of squares
                    if (objArray.length === 2) {
                        let a;
                        let b;
                        a = objArray.pop();
                        b = objArray.pop();

                        if (
                            even(Number(a.power)) &&
                            even(Number(b.power)) &&
                            a.sign() === b.sign() &&
                            a.group === S &&
                            b.group === S
                        ) {
                            throw new Error('Unable to factor');
                        }
                        if (a.isComposite() && /** @type {FracType} */ (b.power).equals(2) && a.sign() !== b.sign()) {
                            // Remove the square from b
                            b = removeSquare(b);
                            const f = __.Factor.factorInner(
                                /** @type {NerdamerSymbolType} */ (_.add(a, separated.constants))
                            );
                            if (/** @type {FracType} */ (f.power).equals(2)) {
                                f.toLinear();
                                factors.add(/** @type {NerdamerSymbolType} */ (_.subtract(f.clone(), b.clone())));
                                factors.add(/** @type {NerdamerSymbolType} */ (_.add(f, b)));
                                symbol = new NerdamerSymbol(1);
                            }
                        } else {
                            a = a.powSimp();
                            b = b.powSimp();

                            if (
                                (a.group === S || a.fname === '') &&
                                a.power.equals(2) &&
                                (b.group === S || b.fname === '') &&
                                b.power.equals(2) &&
                                !separated.constants
                            ) {
                                if (a.multiplier.lessThan(0)) {
                                    const t = b;
                                    b = a;
                                    a = t;
                                }
                                if (a.multiplier.greaterThan(0)) {
                                    a = removeSquare(a);
                                    b = removeSquare(b);
                                }

                                factors.add(/** @type {NerdamerSymbolType} */ (_.subtract(a.clone(), b.clone())));
                                factors.add(/** @type {NerdamerSymbolType} */ (_.add(a, b)));
                                symbol = new NerdamerSymbol(1);
                            }
                        }
                    }
                } catch (e) {
                    if (e.message === 'timeout') {
                        throw e;
                    }
                }

                return symbol;
            },
            // Factoring for multivariate
            /**
             * Factoring for multivariate polynomials
             *
             * @param {NerdamerSymbolType} symbol
             * @param {FactorsLike} factors
             * @returns {NerdamerSymbolType}
             */
            mfactor(symbol, factors) {
                if (symbol.group === FN) {
                    if (symbol.fname === 'sqrt') {
                        const factors2 = new Factors();
                        let arg = __.Factor.common(symbol.args[0].clone(), factors2);
                        arg = __.Factor.coeffFactor(arg, null);
                        symbol = /** @type {NerdamerSymbolType} */ (
                            _.multiply(_.symfunction('sqrt', [arg]), _.parse(symbol.multiplier))
                        );
                        factors2.each(x => {
                            symbol = /** @type {NerdamerSymbolType} */ (
                                _.multiply(symbol, _.parse(core.Utils.format('sqrt({0})', String(x))))
                            );
                        });
                    } else {
                        factors.add(symbol);
                        symbol = new NerdamerSymbol(1);
                    }
                } else {
                    // Square free factorization
                    symbol = __.Factor.mSqfrFactor(symbol, factors);

                    // Try factor out common factors
                    // symbol = __.Factor.common(symbol, factors);

                    const vars = variables(symbol);
                    const symbols = /** @type {NerdamerSymbolType[]} */ (
                        symbol
                            .collectSymbols()
                            .map(x => NerdamerSymbol.unwrapSQRT(/** @type {NerdamerSymbolType} */ (x)))
                    );
                    const sorted = {};
                    const maxes = {};
                    const l = vars.length;
                    const n = symbols.length;
                    // Take all the variables in the symbol and organize by variable name
                    // e.g. a^2+a^2+b*a -> {a: {a^3, a^2, b*a}, b: {b*a}}

                    for (let i = 0; i < l; i++) {
                        const v = vars[i];
                        sorted[v] = new NerdamerSymbol(0);
                        for (let j = 0; j < n; j++) {
                            const s = symbols[j];
                            if (s.contains(v)) {
                                const p =
                                    s.value === v
                                        ? /** @type {FracType} */ (s.power).toDecimal()
                                        : /** @type {FracType} */ (s.symbols[v].power).toDecimal();
                                if (!maxes[v] || p < maxes[v]) {
                                    maxes[v] = p;
                                }
                                sorted[v] = /** @type {NerdamerSymbolType} */ (_.add(sorted[v], s.clone()));
                            }
                        }
                    }

                    for (const x in sorted) {
                        if (!Object.hasOwn(sorted, x)) {
                            continue;
                        }
                        const r = /** @type {NerdamerSymbolType} */ (_.parse(`${x}^${maxes[x]}`));
                        const div = /** @type {NerdamerSymbolType} */ (_.divide(sorted[x], r));
                        const newFactor = /** @type {NerdamerSymbolType} */ (_.expand(div));

                        if (newFactor.equals(1) || newFactor.equals(-1)) {
                            break;
                        } // Why divide by one. Just move
                        const divided = __.div(symbol.clone(), newFactor);

                        if (/** @type {NerdamerSymbolType} */ (divided[0]).equals(0)) {
                            // Cant factor anymore
                            break;
                        }

                        // We potentially ended up with fractional coefficients when the
                        // trial division was performed. We need to remove
                        // This check will more then likely become superfluous with improvements
                        // to polynomial division
                        if (/** @type {NerdamerSymbolType} */ (divided[1]).equals(0)) {
                            let hasFractions = false;

                            /** @type {NerdamerSymbolType} */ (divided[0]).each(elem => {
                                if (!isInt(elem.multiplier)) {
                                    hasFractions = true;
                                }
                            });

                            // The factor isn't really a factor and needs to be put back
                            if (hasFractions) {
                                divided[1] = /** @type {NerdamerSymbolType} */ (
                                    _.expand(_.multiply(divided[1], newFactor))
                                );
                                // Since the new factor is not just one, we exit.
                                break;
                            }
                        }

                        const negNumericFactor =
                            isInt(newFactor) && /** @type {NerdamerSymbolType} */ (newFactor).lessThan(0);

                        if (/** @type {NerdamerSymbolType} */ (divided[1]).equals(0) && !negNumericFactor) {
                            // We found at least one factor

                            // factors.add(newFactor);
                            const d = __.divWithCheck(
                                symbol.clone(),
                                /** @type {NerdamerSymbolType} */ (divided[0]).clone()
                            );
                            const innerR = /** @type {NerdamerSymbolType} */ (d[0]);

                            // Nothing left to do since we didn't get a reduction
                            if (innerR.equals(0)) {
                                return symbol;
                            }

                            symbol = /** @type {NerdamerSymbolType} */ (d[1]);
                            // We don't want to just flip the sign. If the remainder is -1 then we accomplished nothing
                            // and we just return the symbol;
                            // If r equals zero then there's nothing left to do so we're done

                            if (innerR.equals(-1) && !symbol.equals(0)) {
                                return symbol;
                            }

                            const factor = /** @type {NerdamerSymbolType} */ (divided[0]);

                            if (symbol.equals(factor)) {
                                const rem = __.Factor.reduce(factor, factors);

                                if (!symbol.equals(rem)) {
                                    return __.Factor.mfactor(rem, factors);
                                }

                                return rem;
                            }
                            factors.add(factor);
                            // If the remainder of the symbol is zero then we're done. TODO: Rethink this logic a bit.
                            if (symbol.equals(0)) {
                                return innerR;
                            }

                            if (innerR.isConstant('all')) {
                                factors.add(innerR);
                                return innerR;
                            }

                            symbol = __.Factor.mfactor(innerR, factors);
                            // // sanitization: eliminate "-(-x)"
                            // for (let xk in symbol.symbols) {
                            //     let x = symbol.symbols[xk];
                            //     if ((x.group === CB || x.group === CP || x.group === PL) &&
                            //         x.multiplier.equals(-1)) {
                            //         console.log("replacing "+x)
                            //         symbol[xk] = _.parse(x);
                            //         console.log("with "+symbol[xk])
                            //     }
                            // }
                            return symbol;
                        }
                    }
                }

                // Difference of squares factorization
                symbol = __.Factor.sqdiff(symbol, factors);

                // Factors by fishing for zeroes
                symbol = __.Factor.zeroes(symbol, factors);

                // // sanitization: eliminate "-(-x)"
                // for (let xk in symbol.symbols) {
                //     let x = symbol.symbols[xk];
                //     if ((x.group === CB || x.group === CP || x.group === PL) &&
                //         x.multiplier.equals(-1)) {
                //         console.log("replacing "+x)
                //         symbol[xk] = _.parse(x);
                //         console.log("with "+symbol[xk])
                //     }
                // }
                return symbol;
            },
        },
        /**
         * Checks to see if a set of "equations" is linear.
         *
         * @param {Array} s - The set of equations to check
         * @returns {boolean}
         */
        allLinear(s) {
            const l = s.length;
            for (let i = 0; i < l; i++) {
                if (!__.isLinear(s[i])) {
                    return false;
                }
            }
            return true;
        },
        /*
         * Checks to see if the "equation" is linear
         * @param {NerdamerSymbolType} e
         * @returns {boolean}
         */
        isLinear(e) {
            let status = false;
            const g = e.group;
            if (g === PL || g === CP) {
                status = true;
                for (const s in e.symbols) {
                    if (!Object.hasOwn(e.symbols, s)) {
                        continue;
                    }
                    const symbol = e.symbols[s];
                    const sg = symbol.group;
                    if (sg === FN || sg === EX) {
                        status = false;
                    }
                    if (sg === CB) {
                        // Needs further checking since it might be imaginary
                        status = variables(symbol).length === 1;
                    } else if (sg === PL || sg === CP) {
                        status = __.isLinear(symbol);
                    } else if (symbol.group !== N && symbol.power.toString() !== '1') {
                        status = false;
                        break;
                    }
                }
            } else if (g === S && /** @type {number} */ (/** @type {unknown} */ (e.power)) === 1) {
                status = true;
            }
            return status;
        },
        gcd(...rest) {
            let args;
            if (rest.length === 1 && rest[0] instanceof core.Vector) {
                args = rest[0].elements;
            } else {
                args = rest;
            }

            // Short-circuit early
            if (args.length === 0) {
                return new NerdamerSymbol(1);
            }
            if (args.length === 1) {
                return args[0];
            }

            let appeared = [];
            let evaluate = false;
            for (let i = 0; i < args.length; i++) {
                const arg = /** @type {NerdamerSymbolType} */ (args[i]);
                if (arg.group === FN && arg.fname === 'gcd') {
                    // Compress gcd(a,gcd(b,c)) into gcd(a,b,c)
                    args = args.concat(arg.args);
                    // Do not keep gcd in args
                    args.splice(i, 1);
                } else {
                    // Look if there are any common variables such that
                    // gcd(a,b) => gcd(a,b); gcd(a,a) => a
                    const vars = variables(arg);
                    if (core.Utils.haveIntersection(vars, appeared)) {
                        // Ok, there are common variables
                        evaluate = true;
                        break;
                    } else {
                        appeared = appeared.concat(vars);
                    }
                }
            }

            // Appeared.length is 0 when all arguments are group N
            if (evaluate || appeared.length === 0) {
                // TODO: distribute exponent so that (a^-1*b^-1)^-1 => a*b
                if (
                    args.every(symbol =>
                        /** @type {NerdamerSymbolType} */ (
                            /** @type {NerdamerSymbolType} */ (symbol).getDenom()
                        ).equals(1)
                    )
                ) {
                    let aggregate = /** @type {NerdamerSymbolType} */ (args[0]);

                    for (let i = 1; i < args.length; i++) {
                        aggregate = /** @type {NerdamerSymbolType} */ (
                            __.gcd_(/** @type {NerdamerSymbolType} */ (args[i]), aggregate)
                        );
                    }
                    return aggregate;
                }
                // Gcd_ cannot handle denominators correctly
                return _.divide(
                    __.gcd.apply(
                        null,
                        /** @type {NerdamerSymbolType[]} */ (
                            args.map(
                                symbol =>
                                    /** @type {NerdamerSymbolType} */ (
                                        /** @type {NerdamerSymbolType} */ (symbol).getNum()
                                    )
                            )
                        )
                    ),
                    __.lcm.apply(
                        null,
                        /** @type {NerdamerSymbolType[]} */ (
                            args.map(
                                symbol =>
                                    /** @type {NerdamerSymbolType} */ (
                                        /** @type {NerdamerSymbolType} */ (symbol).getDenom()
                                    )
                            )
                        )
                    )
                );
            }
            return _.symfunction('gcd', args);
        },
        gcd_(a, b) {
            if (a.group === FN || a.group === P) {
                a = /** @type {NerdamerSymbolType} */ (core.Utils.block('PARSE2NUMBER', () => _.parse(a)));
            }
            if (b.group === FN || b.group === P) {
                b = /** @type {NerdamerSymbolType} */ (core.Utils.block('PARSE2NUMBER', () => _.parse(b)));
            }

            if (b.group === FN) {
                b = /** @type {NerdamerSymbolType} */ (core.Utils.block('PARSE2NUMBER', () => _.parse(b)));
            }

            if (a.isConstant() && b.isConstant()) {
                // Return core.Math2.QGCD(new Frac(Number(a)), new Frac(Number(b)));
                return new NerdamerSymbol(core.Math2.QGCD(new Frac(Number(a)), new Frac(Number(b))));
            }

            const den = /** @type {NerdamerSymbolType} */ (
                _.multiply(
                    /** @type {NerdamerSymbolType} */ (a.getDenom()) || new NerdamerSymbol(1),
                    /** @type {NerdamerSymbolType} */ (b.getDenom()) || new NerdamerSymbol(1)
                )
            ).invert();
            a = /** @type {NerdamerSymbolType} */ (_.multiply(a.clone(), den.clone()));
            b = /** @type {NerdamerSymbolType} */ (_.multiply(b.clone(), den.clone()));

            // Feels counter intuitive but it works. Issue #123 (nerdamer("gcd(x+y,(x+y)^2)"))
            a = /** @type {NerdamerSymbolType} */ (_.expand(a));
            b = /** @type {NerdamerSymbolType} */ (_.expand(b));

            if (a.group === CB || b.group === CB) {
                const q = /** @type {NerdamerSymbolType} */ (_.divide(a.clone(), b.clone())); // Get the quotient
                const t = /** @type {NerdamerSymbolType} */ (_.multiply(b.clone(), q.getDenom().invert())); // Multiply by the denominator
                // if they have a common factor then the result will not equal one
                if (!t.equals(1)) {
                    return t;
                }
            }

            // Just take the gcd of each component when either of them is in group EX
            if (a.group === EX || b.group === EX) {
                const gcdM = new NerdamerSymbol(core.Math2.QGCD(a.multiplier, b.multiplier));
                const gcdV = __.gcd_(
                    a.value === CONST_HASH
                        ? new NerdamerSymbol(1)
                        : /** @type {NerdamerSymbolType} */ (_.parse(a.value)),
                    b.value === CONST_HASH
                        ? new NerdamerSymbol(1)
                        : /** @type {NerdamerSymbolType} */ (_.parse(b.value))
                );
                const gcdP = __.gcd_(
                    /** @type {NerdamerSymbolType} */ (_.parse(a.power)),
                    /** @type {NerdamerSymbolType} */ (_.parse(b.power))
                );
                return _.multiply(gcdM, _.pow(gcdV, gcdP));
            }

            if (a.length < b.length) {
                // Swap'm
                const t = a;
                a = b;
                b = t;
            }
            const varsA = variables(a);
            const varsB = variables(b);

            // GCD of a polynomial and a constant: gcd(poly, const) = gcd of coefficients with const
            // For symbolic variables, gcd(a, 1) = 1, gcd(a, 0) = a
            if ((varsA.length === 1 && varsB.length === 0) || (varsA.length === 0 && varsB.length === 1)) {
                // One is a variable/polynomial, one is a constant
                const polySymbol = varsA.length === 1 ? a : b;
                const constSymbol = varsA.length === 0 ? a : b;

                if (constSymbol.equals(0)) {
                    return polySymbol;
                }
                // GCD of polynomial with non-zero constant
                // For symbolic case, this is just the gcd of coefficients
                return new NerdamerSymbol(core.Math2.QGCD(polySymbol.multiplier, constSymbol.multiplier));
            }

            if (varsA.length === varsB.length && varsA.length === 1 && varsA[0] === varsB[0]) {
                const polyA = new Polynomial(a);
                const polyB = new Polynomial(b);
                return _.divide(polyA.gcd(polyB).toSymbol(), den);
            }
            // Get the gcd of the multipiers
            // get rid of gcd in coeffs
            const multipliers = [];
            a.each(x => {
                multipliers.push(x.multiplier);
            });
            b.each(x => {
                multipliers.push(x.multiplier);
            });

            let T;
            while (!b.equals(0)) {
                const t = b.clone();
                a = a.clone();
                T = __.div(a, t);

                b = /** @type {NerdamerSymbolType} */ (T[1]);
                if (/** @type {NerdamerSymbolType} */ (T[0]).equals(0)) {
                    // Return _.multiply(new NerdamerSymbol(core.Math2.QGCD(a.multiplier, b.multiplier)), b);
                    return _.divide(new NerdamerSymbol(core.Math2.QGCD(a.multiplier, b.multiplier)), den);
                }
                a = t;
            }

            const gcd = core.Math2.QGCD.apply(undefined, multipliers);

            if (!gcd.equals(1)) {
                a.each(x => {
                    x.multiplier = x.multiplier.divide(gcd);
                });
            }

            // Return symbolic function for gcd in indeterminate form
            if (a.equals(1) && !a.isConstant() && !b.isConstant()) {
                return _.divide(_.symfunction('gcd', [a, b]), den);
            }

            return _.divide(a, den);
        },
        lcm(...rest) {
            // https://math.stackexchange.com/a/319310
            // generalization of the 2-variable formula of lcm

            let args;
            if (rest.length === 1) {
                if (rest[0] instanceof core.Vector) {
                    args = rest[0].elements;
                } else {
                    _.error('lcm expects either 1 vector or 2 or more arguments');
                }
            } else {
                args = rest;
            }

            // Product of all arguments
            // start with new NerdamerSymbol(1) so that prev.clone() which makes unnessesary clones can be avoided
            const numer = args.reduce((prev, curr) => _.multiply(prev, curr.clone()), new NerdamerSymbol(1));

            // Gcd of complementary terms
            const denomArgs =
                // https://stackoverflow.com/a/18223072
                // take all complementary terms, e.g.
                // [a,b,c] => [a*b, b*c, a*c]
                // [a,b,c,d] => [a*b*c, a*b*d, a*c*d, b*c*d]
                /** @type {NerdamerSymbolType[]} */ (
                    (function generateComplementTerms(input, size) {
                        size = Number(size);
                        const results = [];
                        let result;
                        let mask;
                        let i;
                        const total = 2 ** input.length;
                        for (mask = size; mask < total; mask++) {
                            result = [];
                            i = input.length - 1;

                            do {
                                // eslint-disable-next-line no-bitwise -- Bit masking for combinatorial generation
                                if ((mask & (1 << i)) !== 0) {
                                    result.push(input[i]);
                                }
                            } while (i--);

                            if (result.length === size) {
                                results.push(result);
                            }
                        }
                        return results;
                        // Start with new NerdamerSymbol(1) so that prev.clone() which makes unnessesary clones can be avoided
                    })(args, args.length - 1).map(x =>
                        x.reduce(
                            (prev, curr) => /** @type {NerdamerSymbolType} */ (_.multiply(prev, curr.clone())),
                            new NerdamerSymbol(1)
                        )
                    )
                );

            let denom;
            // Don't eat the gcd term if all arguments are symbols
            if (args.every(x => core.Utils.isVariableSymbol(x))) {
                denom = _.symfunction('gcd', core.Utils.arrayUnique(denomArgs));
            } else {
                denom = __.gcd.apply(
                    null,
                    /** @type {[NerdamerSymbolType, NerdamerSymbolType, ...NerdamerSymbolType[]]} */ (denomArgs)
                );
            }
            // Divide product of all arguments by gcd of complementary terms
            const div = _.divide(numer, denom);
            return div;
        },
        /**
         * Divides one expression by another
         *
         * @param {NerdamerSymbolType} symbol1
         * @param {NerdamerSymbolType} symbol2
         * @returns {NerdamerSymbolType}
         */
        divide(symbol1, symbol2) {
            let den;
            const factored = /** @type {NerdamerSymbolType} */ (__.Factor.factorInner(symbol1.clone()));
            den = factored.getDenom();
            if (den.isConstant('all')) {
                // Reset the denominator since we're not dividing by it anymore
                den = new NerdamerSymbol(1);
            } else {
                symbol1 = /** @type {NerdamerSymbolType} */ (
                    _.expand(
                        NerdamerSymbol.unwrapPARENS(
                            /** @type {NerdamerSymbolType} */ (_.multiply(factored, den.clone()))
                        )
                    )
                );
            }
            const result = __.div(symbol1, symbol2);
            const remainder = /** @type {NerdamerSymbolType} */ (_.divide(result[1], symbol2));
            return /** @type {NerdamerSymbolType} */ (
                _.divide(/** @type {NerdamerSymbolType} */ (_.add(result[0], remainder)), den)
            );
        },
        divWithCheck(symbol1, symbol2) {
            const fail = [new NerdamerSymbol(0), symbol1.clone()];
            const div = __.div(symbol1, symbol2);
            // GM safety check because __.div() produces b.s. sometimes
            // see whether multiplication comes out clean
            const a = symbol1.clone();
            let b = /** @type {NerdamerSymbolType} */ (_.multiply(div[0].clone(), symbol2.clone()));
            b = /** @type {NerdamerSymbolType} */ (_.add(b, div[1].clone()));
            let test = /** @type {NerdamerSymbolType} */ (_.subtract(a, b));
            test = /** @type {NerdamerSymbolType} */ (_.expand(test));
            // Test = __.Simplify._simplify(test);

            if (test.equals(0)) {
                // Ok, seems good
                return div;
            }
            // False alarm, get the default back
            // console.log("nerdamer-prime: div failed: " + test);
            return fail;
        },
        div(symbol1, symbol2) {
            // If all else fails then assume that division failed with
            // a remainder of zero and the original quotient
            const fail = [new NerdamerSymbol(0), symbol1.clone()];

            try {
                // Division by constants
                if (symbol2.isConstant('all')) {
                    symbol1.each(x => {
                        x.multiplier = x.multiplier.divide(symbol2.multiplier);
                    });
                    return [symbol1, new NerdamerSymbol(0)];
                }
                // So that factorized symbols don't affect the result
                symbol1 = /** @type {NerdamerSymbolType} */ (_.expand(symbol1));
                symbol2 = /** @type {NerdamerSymbolType} */ (_.expand(symbol2));
                // Special case. May need revisiting
                if (symbol1.group === S && symbol2.group === CP) {
                    const x = symbol1.value;
                    const f = /** @type {DecomposeResultType} */ (core.Utils.decompose_fn(symbol2.clone(), x, true));
                    if (symbol1.isLinear() && f.x && f.x.isLinear() && symbol2.isLinear()) {
                        const k = NerdamerSymbol.create(symbol1.multiplier);
                        return [
                            /** @type {NerdamerSymbolType} */ (_.divide(k.clone(), f.a.clone())),
                            /** @type {NerdamerSymbolType} */ (_.divide(_.multiply(k, f.b), f.a)).negate(),
                        ];
                    }
                }
                if (symbol1.group === S && symbol2.group === S) {
                    const r = /** @type {NerdamerSymbolType} */ (_.divide(symbol1.clone(), symbol2.clone()));
                    if (r.isConstant()) // We have a whole
                    {
                        return [r, new NerdamerSymbol(0)];
                    }
                    return [new NerdamerSymbol(0), symbol1.clone()];
                }
                const symbol1HasFunc = symbol1.hasFunc();
                const symbol2HasFunc = symbol2.hasFunc();
                let parseFuncs = false;
                let subs;

                // Substitute out functions so we can treat them as regular variables
                if (symbol1HasFunc || symbol2HasFunc) {
                    parseFuncs = true;
                    /** @type {Record<string, string>} */
                    const map = {};
                    symbol1 = /** @type {NerdamerSymbolType} */ (_.parse(core.Utils.subFunctions(symbol1, map)));
                    symbol2 = /** @type {NerdamerSymbolType} */ (_.parse(core.Utils.subFunctions(symbol2, map)));
                    subs = core.Utils.getFunctionsSubs(map);
                }
                // Get a list of the variables
                const vars = core.Utils.arrayUnique(variables(symbol1).concat(variables(symbol2)));
                let quot;
                let rem;
                let den;

                // Treat imaginary numbers as variables
                if (symbol1.isImaginary() || symbol2.isImaginary()) {
                    vars.push(core.Settings.IMAGINARY);
                }

                if (vars.length === 1) {
                    const q = new Polynomial(symbol1).divide(new Polynomial(symbol2));
                    quot = q[0].toSymbol();
                    rem = q[1].toSymbol();
                } else {
                    vars.push(CONST_HASH); // This is for the numbers
                    const reconvert = function (arr) {
                        let symbol = new NerdamerSymbol(0);
                        for (let i = 0; i < arr.length; i++) {
                            const x = arr[i].toSymbol();
                            symbol = /** @type {NerdamerSymbolType} */ (_.add(symbol, x));
                        }
                        return symbol;
                    };

                    // Silly Martin. This is why you document. I don't remember now
                    const getUniqueMax = function (term, any) {
                        const max = Math.max.apply(null, term.terms);
                        let count = 0;
                        let idx;

                        if (!any) {
                            for (let i = 0; i < term.terms.length; i++) {
                                if (term.terms[i].equals(max)) {
                                    idx = i;
                                    count++;
                                }
                                if (count > 1) {
                                    return undefined;
                                }
                            }
                        }
                        if (any) {
                            for (let i = 0; i < term.terms.length; i++) {
                                if (term.terms[i].equals(max)) {
                                    idx = i;
                                    break;
                                }
                            }
                        }
                        return [max, idx, term];
                    };

                    const tMap = core.Utils.toMapObj(vars);
                    const initSort = function (a, b) {
                        return b.sum.subtract(a.sum);
                    };

                    const s1 = symbol1.tBase(tMap).sort(initSort);
                    const s2 = symbol2.tBase(tMap).sort(initSort);

                    // Tries to find an LT in the dividend that will satisfy division
                    const getDet = function (s, lookat) {
                        lookat ||= 0;
                        const det = s[lookat];
                        const l = s.length;
                        if (!det) {
                            return undefined;
                        }
                        // Eliminate the first term if it doesn't apply
                        let umax = getUniqueMax(det);
                        for (let i = lookat + 1; i < l; i++) {
                            const term = s[i];
                            const isEqual = det.sum.equals(term.sum);
                            if (!isEqual && umax) {
                                break;
                            }
                            if (isEqual) {
                                // Check the differences of their maxes. The one with the biggest difference governs
                                // e.g. x^2*y^3 vs x^2*y^3 is unclear but this isn't the case in x*y and x^2
                                let max1;
                                let max2;
                                let idx1;
                                let idx2;
                                const l2 = det.terms.length;
                                for (let j = 0; j < l2; j++) {
                                    const item1 = det.terms[j];
                                    const item2 = term.terms[j];
                                    if (typeof max1 === 'undefined' || item1.greaterThan(max1)) {
                                        max1 = item1;
                                        idx1 = j;
                                    }
                                    if (typeof max2 === 'undefined' || item2.greaterThan(max2)) {
                                        max2 = item2;
                                        idx2 = j;
                                    }
                                }
                                // Check their differences
                                const d1 = max1.subtract(term.terms[idx1]);
                                const d2 = max2.subtract(det.terms[idx2]);
                                if (d2 > d1) {
                                    umax = [max2, idx2, term];
                                    break;
                                }
                                if (d1 > d2) {
                                    umax = [max1, idx1, det];
                                    break;
                                }
                            } else {
                                // Check if it's a suitable pick to determine the order
                                umax = getUniqueMax(term);
                                // If(umax) return umax;
                                if (umax) {
                                    break;
                                }
                            }
                            umax = getUniqueMax(term); // Calculate a new unique max
                        }

                        // If still no umax then any will do since we have a tie
                        if (!umax) {
                            return getUniqueMax(s[0], true);
                        }
                        let e;
                        let idx;
                        for (let i = 0; i < s2.length; i++) {
                            const cterm = s2[i].terms;
                            // Confirm that this is a good match for the denominator
                            idx = umax[1];
                            if (idx === cterm.length - 1) {
                                return undefined;
                            }
                            e = cterm[idx];
                            if (!e.equals(0)) {
                                break;
                            }
                        }
                        if (e.equals(0)) {
                            return getDet(s, ++lookat);
                        } // Look at the next term

                        return umax;
                    };

                    const isLarger = function (a, b) {
                        if (!a || !b) {
                            return false;
                        } // It's empty so...
                        for (let i = 0; i < a.terms.length; i++) {
                            if (a.terms[i].lessThan(b.terms[i])) {
                                return false;
                            }
                        }
                        return true;
                    };

                    const target = isLarger(s1[0], s2[0]) && s1[0].count > s2[0].count ? s2 : s1; // Since the num is already larger than we can get the det from denom
                    const det = getDet(target); // We'll begin by assuming that this will let us know which term
                    const quotient = [];
                    if (det) {
                        let leadVar = det[1];
                        const canDivide = function (a, b) {
                            if (a[0].sum.equals(b[0].sum)) {
                                return a.length >= b.length;
                            }
                            return true;
                        };

                        const tryBetterLeadVar = function (sym1, sym2, leadVarParam) {
                            const checked = [];
                            for (let i = 0; i < sym1.length; i++) {
                                const t = sym1[i];
                                for (let j = 0; j < t.terms.length; j++) {
                                    const cf = checked[j];
                                    const tt = t.terms[j];
                                    if (i === 0) {
                                        checked[j] = tt;
                                    } // Add the terms for the first one
                                    else if (cf && !cf.equals(tt)) {
                                        checked[j] = undefined;
                                    }
                                }
                            }
                            for (let i = 0; i < checked.length; i++) {
                                const t = checked[i];
                                if (t && !t.equals(0)) {
                                    return i;
                                }
                            }
                            return leadVarParam;
                        };
                        const sf = function (a, b) {
                            const l1 = a.len();
                            const l2 = b.len();
                            const blv = b.terms[leadVar];
                            const alv = a.terms[leadVar];
                            if (l2 > l1 && blv.greaterThan(alv)) {
                                return l2 - l1;
                            }
                            return blv.subtract(alv);
                        };

                        // Check to see if there's a better leadVar
                        leadVar = tryBetterLeadVar(s1, s2, leadVar);
                        // Reorder both according to the max power
                        s1.sort(sf); // Sort them both according to the leading variable power
                        s2.sort(sf);

                        // Try to adjust if den is larger
                        const fdt = s2[0];
                        const fnt = s1[0];

                        den = new MVTerm(new Frac(1), [], fnt.map);
                        if (fdt.sum.greaterThan(fnt.sum) && fnt.len() > 1) {
                            for (let i = 0; i < fnt.terms.length; i++) {
                                const d = fdt.terms[i].subtract(fnt.terms[i]);
                                if (d.equals(0)) {
                                    den.terms[i] = new Frac(0);
                                } else {
                                    const nd = d.add(new Frac(1));
                                    den.terms[i] = d;
                                    for (let j = 0; j < s1.length; j++) {
                                        s1[j].terms[i] = s1[j].terms[i].add(nd);
                                    }
                                }
                            }
                        }

                        let dividendLarger = isLarger(s1[0], s2[0]);

                        let safety = 0;
                        const max = 200;

                        while (dividendLarger && canDivide(s1, s2)) {
                            if (safety++ > max) {
                                throw new core.exceptions.InfiniteLoopError('Unable to compute!');
                            }

                            const q = s1[0].divide(s2[0]);

                            quotient.push(q); // Add what's divided to the quotient
                            s1.shift(); // The first one is guaranteed to be gone so remove from dividend
                            for (let i = 1; i < s2.length; i++) {
                                // Loop through the denominator
                                const t = s2[i].multiply(q).generateImage();
                                const l2 = s1.length;
                                // If we're subtracting from 0
                                if (l2 === 0) {
                                    t.coeff = t.coeff.neg();
                                    s1.push(t);
                                    s1.sort(sf);
                                }

                                for (let j = 0; j < l2; j++) {
                                    const cur = s1[j];
                                    if (cur.getImg() === t.getImg()) {
                                        cur.coeff = cur.coeff.subtract(t.coeff);
                                        if (cur.coeff.equals(0)) {
                                            core.Utils.remove(s1, j);
                                            j--; // Adjust the iterator
                                        }
                                        break;
                                    }
                                    if (j === l2 - 1) {
                                        t.coeff = t.coeff.neg();
                                        s1.push(t);
                                        s1.sort(sf);
                                    }
                                }
                            }
                            dividendLarger = isLarger(s1[0], s2[0]);

                            if (!dividendLarger && s1.length >= s2.length) {
                                // One more try since there might be a terms that is larger than the LT of the divisor
                                for (let i = 1; i < s1.length; i++) {
                                    dividendLarger = isLarger(s1[i], s2[0]);
                                    if (dividendLarger) {
                                        // Take it from its current position and move it to the front
                                        s1.unshift(core.Utils.remove(s1, i));
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    quot = reconvert(quotient);
                    rem = reconvert(s1);

                    if (typeof den !== 'undefined') {
                        den = den.toSymbol();
                        quot = _.divide(quot, den.clone());
                        rem = _.divide(rem, den);
                    }
                }

                // Put back the functions
                if (parseFuncs) {
                    quot = _.parse(quot.text(), subs);
                    rem = _.parse(rem.text(), subs);
                }

                return [quot, rem];
            } catch (e) {
                if (e.message === 'timeout') {
                    throw e;
                }
                return fail;
            }
        },
        line(v1, v2, x) {
            if (core.Utils.isArray(v1)) {
                v1 = core.Utils.convertToVector(/** @type {ExpressionParam[]} */ (v1));
            }
            if (core.Utils.isArray(v2)) {
                v2 = core.Utils.convertToVector(/** @type {ExpressionParam[]} */ (v2));
            }
            const xVar = /** @type {NerdamerSymbolType} */ (_.parse(x || 'x'));
            if (!core.Utils.isVector(v1) || !core.Utils.isVector(v2)) {
                _.error(`Line expects a vector! Received "${v1}" & "${v2}"`);
            }
            const vec1 = /** @type {VectorType} */ (v1);
            const vec2 = /** @type {VectorType} */ (v2);
            const dx = _.subtract(
                /** @type {NerdamerSymbolType} */ (vec2.e(1)).clone(),
                /** @type {NerdamerSymbolType} */ (vec1.e(1)).clone()
            );
            const dy = _.subtract(
                /** @type {NerdamerSymbolType} */ (vec2.e(2)).clone(),
                /** @type {NerdamerSymbolType} */ (vec1.e(2)).clone()
            );
            const m = _.divide(dy, dx);
            const a = _.multiply(xVar, /** @type {NerdamerSymbolType} */ (m).clone());
            const b = _.multiply(/** @type {NerdamerSymbolType} */ (vec1.e(1)).clone(), m);
            return _.add(_.subtract(a, b), /** @type {NerdamerSymbolType} */ (vec1.e(2)).clone());
        },
        PartFrac: {
            /**
             * Creates a template for partial fraction decomposition
             *
             * @param {NerdamerSymbolType} den - The denominator
             * @param {NerdamerSymbolType} denomFactors - Factored form of denominator
             * @param {NerdamerSymbolType[]} fArray - Array to collect factor components
             * @param {NerdamerSymbolType} v - The variable
             * @returns {[NerdamerSymbolType[], (NerdamerSymbolType | VectorType | MatrixType)[], number[]]}
             */
            createTemplate(den, denomFactors, fArray, v) {
                // Clean up the denominator function by factors so it reduces nicely
                den = __.Factor.factorInner(den);

                // Clean up factors. This is so inefficient but factors are wrapped in parens for safety
                den.each((x, key) => {
                    if (x.group === FN && x.fname === '' && x.args[0].group === S) {
                        const y = x.args[0];
                        if (den.symbols) {
                            delete den.symbols[key];
                            den.symbols[y.value] = y;
                        } else {
                            den = x.args[0];
                        }
                    }
                });

                let f;
                let p;
                let deg;
                const factors = /** @type {NerdamerSymbolType[]} */ (denomFactors.collectFactors?.() || []);
                const factorsVec = []; // A vector for the template
                const degrees = [];
                const m = new NerdamerSymbol(1);

                for (let i = 0; i < factors.length; i++) {
                    // Loop through the factors
                    const factor = NerdamerSymbol.unwrapPARENS(factors[i]);
                    // If in he for P^n where P is polynomial and n = integer
                    if (factor.power.greaterThan(1)) {
                        p = Number(factor.power);
                        f = factor.clone().toLinear(); // Remove the power so we have only the function
                        deg = Number(__.degree(f, v)); // Get the degree of f
                        // expand the factor
                        for (let j = 0; j < p; j++) {
                            const efactor = /** @type {NerdamerSymbolType} */ (
                                _.pow(f.clone(), new NerdamerSymbol(j + 1))
                            );
                            fArray.push(efactor.clone());
                            const d = _.divide(den.clone(), efactor.clone());
                            degrees.push(deg);
                            factorsVec.push(d);
                        }
                    } else {
                        /*
                     Possible bug.
                     Removed: causes 1/(20+24*x+4*x^2) to result in (-1/64)*(5+x)^(-1)+(1/64)*(1+x)^(-1)
                     else if(factor.isConstant('all')) {
                     m = _.multiply(m, factor);
                     }
                     */
                        // get the degree of the factor so we tack it on tot he factor. This should probably be an array
                        // but for now we note it on the symbol
                        deg = Number(__.degree(factor, v));
                        fArray.push(factor);
                        let d = _.divide(den.clone(), factor.clone());
                        d = /** @type {NerdamerSymbolType} */ (
                            _.expand(NerdamerSymbol.unwrapPARENS(/** @type {NerdamerSymbolType} */ (d)))
                        );
                        degrees.push(deg);
                        factorsVec.push(d);
                    }
                }
                // Put back the constant
                fArray = /** @type {NerdamerSymbolType[]} */ (fArray.map(x => _.multiply(x, m.clone())));
                return [fArray, factorsVec, degrees];
            },
            /**
             * Performs partial fraction decomposition
             *
             * @param {NerdamerSymbolType} symbol - The expression to decompose
             * @param {NerdamerSymbolType} [v] - The variable
             * @param {boolean} [asArray] - Whether to return as array
             * @returns {NerdamerSymbolType | NerdamerSymbolType[] | VectorType | MatrixType}
             */
            partfrac(symbol, v, asArray) {
                const vars = variables(symbol);

                v ||= /** @type {NerdamerSymbolType} */ (_.parse(vars[0])); // Make wrt optional and assume first variable
                try {
                    let nterms;
                    let div;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let r;
                    let num = /** @type {NerdamerSymbolType} */ (_.expand(symbol.getNum()));
                    const den = /** @type {NerdamerSymbolType} */ (_.expand(symbol.getDenom().toUnitMultiplier()));
                    // Move the entire multipier to the numerator
                    num.multiplier = symbol.multiplier;
                    // We only have a meaningful change if n factors > 1. This means that
                    // the returned group will be a CB
                    // collect the terms wrt the x
                    const vValue = v.value;
                    nterms = num.groupTerms(vValue);
                    // Divide out wholes if top is larger
                    if (Number(__.degree(num, v)) >= Number(__.degree(den, v))) {
                        div = __.div(num.clone(), /** @type {NerdamerSymbolType} */ (_.expand(den.clone())));
                        r = /** @type {NerdamerSymbolType} */ (div[0]); // Remove the wholes
                        num = /** @type {NerdamerSymbolType} */ (div[1]); // Work with the remainder
                        nterms = num.groupTerms(vValue); // Recalculate the nterms
                    } else {
                        r = new NerdamerSymbol(0);
                    }

                    if (Number(__.degree(den, v)) === 1) {
                        const q = /** @type {NerdamerSymbolType} */ (_.divide(num, den));
                        if (asArray) {
                            return [r, q];
                        }
                        return _.add(r, q);
                    }
                    // First factor the denominator. This means that the strength of this
                    // algorithm depends on how well we can factor the denominator.
                    const ofactors = __.Factor.factorInner(den);
                    // Create the template. This method will create the template for solving
                    // the partial fractions. So given x/(x-1)^2 the template creates A/(x-1)+B/(x-1)^2
                    const template = __.PartFrac.createTemplate(den.clone(), ofactors, [], v);
                    const tfactors = template[0]; // Grab the factors
                    const factorsVec = template[1]; // Grab the factor vectors
                    const degrees = template[2]; // Grab the degrees
                    // make note of the powers of each term
                    /** @type {number[]} */
                    const powers = [nterms.length];
                    // Create the dterms vector
                    /** @type {NerdamerSymbolType[][]} */
                    const dterms = [];
                    /** @type {NerdamerSymbolType[]} */
                    const factors = [];
                    /** @type {NerdamerSymbolType[]} */
                    const ks = [];
                    /** @type {NerdamerSymbolType} */
                    let factor;
                    /** @type {number} */
                    let deg;
                    factorsVec.forEach((x, idx) => {
                        factor = tfactors[idx];
                        deg = degrees[idx];
                        for (let i = 0; i < deg; i++) {
                            factors.push(factor.clone());
                            const k = NerdamerSymbol.create(vValue, i);
                            const t = /** @type {NerdamerSymbolType} */ (
                                _.expand(/** @type {NerdamerSymbolType} */ (_.multiply(x, k.clone())))
                            ).groupTerms(vValue);
                            // Make a note of the power which corresponds to the length of the array
                            const p = t.length;
                            powers.push(p);
                            dterms.push(t);
                            ks.push(k.clone());
                        }
                    });
                    // Get the max power
                    const max = core.Utils.arrayMax(/** @type {number[]} */ (powers));

                    // Fill the holes and create a matrix
                    const c = new core.Matrix(core.Utils.fillHoles(nterms, max)).transpose();
                    // For each of the factors we do the same
                    const M = new core.Matrix();
                    for (let i = 0; i < dterms.length; i++) {
                        M.elements.push(core.Utils.fillHoles(dterms[i], max));
                    }

                    // Solve the system of equations
                    const partials = /** @type {MatrixType} */ (_.multiply(M.transpose().invert(), c));
                    // The results are backwards to reverse it
                    // partials.elements.reverse();
                    // convert it all back
                    if (asArray) {
                        /** @type {(NerdamerSymbolType | VectorType | MatrixType)[]} */
                        const retval = [r];
                        partials.each((e, i) => {
                            const term = _.multiply(ks[i], _.divide(e, factors[i]));
                            retval.push(term);
                        });
                        return /** @type {NerdamerSymbolType[]} */ (retval);
                    }
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let retval = r;
                    partials.each((e, i) => {
                        const term = _.multiply(ks[i], _.divide(e, factors[i]));
                        retval = _.add(retval, term);
                    });
                    return retval;
                } catch (e) {
                    if (e.message === 'timeout') {
                        throw e;
                    }
                    // Try to group symbols
                    try {
                        if (symbol.isComposite()) {
                            // Group denominators
                            const denominators = {};

                            symbol.each(x => {
                                const d = x.getDenom();
                                const n = x.getNum();
                                const existing = denominators[d];
                                denominators[d] = existing ? _.add(existing, n) : n;
                            });

                            let t = new NerdamerSymbol(0);

                            for (const x in denominators) {
                                if (!Object.hasOwn(denominators, x)) {
                                    continue;
                                }
                                t = /** @type {NerdamerSymbolType} */ (_.add(t, _.divide(denominators[x], _.parse(x))));
                            }

                            symbol = t;
                        }
                    } catch (e2) {
                        if (e2.message === 'timeout') {
                            throw e2;
                        }
                    }
                }
                return symbol;
            },
        },
        /**
         * Computes the degree of a polynomial
         *
         * @param {NerdamerSymbolType} symbol - The polynomial
         * @param {NerdamerSymbolType} [v] - The variable
         * @param {{ nd: NerdamerSymbolType[]; sd: (NerdamerSymbolType | FracType)[]; depth: number }} [o] - Options for
         *   tracking
         * @returns {NerdamerSymbolType}
         */
        degree(symbol, v, o) {
            o ||= {
                nd: [], // Numeric degrees (stored as NerdamerSymbol)
                sd: [], // Symbolic degrees
                depth: 0, // Call depth
            };

            if (!v) {
                const vars = variables(symbol);
                // The user must specify the variable for multivariate
                if (vars.length > 1) {
                    throw new Error('You must specify the variable for multivariate polynomials!');
                }
                // If it's empty then we're dealing with a constant
                if (vars.length === 0) {
                    return new NerdamerSymbol(0);
                }
                // Assume the variable for univariate
                v = _.parse(vars[0]);
            }

            // Store the group
            const g = symbol.group;
            // We're going to trust the user and assume no EX. Calling isPoly
            // would eliminate this but no sense in checking twice.
            if (symbol.isComposite()) {
                symbol = symbol.clone();
                symbol.distributeExponent();
                symbol.each(x => {
                    o.depth++; // Mark a depth increase
                    __.degree(x, v, o);
                    o.depth--; // We're back
                });
            } else if (symbol.group === CB) {
                symbol.each(x => {
                    o.depth++;
                    __.degree(x, v, o);
                    o.depth++;
                });
            } else if (g === EX && symbol.value === v.value) {
                o.sd.push(symbol.power.clone());
            } else if (g === S && symbol.value === v.value) {
                o.nd.push(/** @type {NerdamerSymbolType} */ (_.parse(symbol.power)));
            } else {
                o.nd.push(new NerdamerSymbol(0));
            }

            // Get the max out of the array - arrayMax uses valueOf() on each symbol to compare numerically
            /** @type {number | undefined} */
            const deg =
                o.nd.length > 0
                    ? core.Utils.arrayMax(/** @type {number[]} */ (/** @type {unknown} */ (o.nd)))
                    : undefined;

            if (o.depth === 0 && o.sd.length > 0) {
                if (deg !== undefined) {
                    // Convert numeric deg back to symbol for the max function
                    o.sd.unshift(/** @type {NerdamerSymbolType} */ (_.parse(deg)));
                }
                return /** @type {NerdamerSymbolType} */ (
                    _.symfunction('max', /** @type {NerdamerSymbolType[]} */ (o.sd))
                );
            }
            // Convert numeric degree to symbol
            return /** @type {NerdamerSymbolType} */ (_.parse(deg ?? 0));
        },
        /**
         * Attempts to complete the square of a polynomial
         *
         * @param {NerdamerSymbolType} symbol
         * @param {string | NerdamerSymbolType} v - The variable to complete the square with respect to
         * @param {boolean} raw
         * @returns {object | NerdamerSymbol[]}
         * @throws {Error}
         */
        sqComplete(symbol, v, raw) {
            if (!core.Utils.isSymbol(v)) {
                v = _.parse(v);
            }
            const stop = function (msg) {
                msg ||= 'Stopping';
                throw new core.exceptions.ValueLimitExceededError(msg);
            };
            // If not CP then nothing to do
            if (!symbol.isPoly(true)) {
                stop('Must be a polynomial!');
            }

            // Declare vars
            const br = core.Utils.inBrackets;
            // Make a copy
            symbol = symbol.clone();
            const deg = core.Algebra.degree(symbol, v); // Get the degree of polynomial
            // must be in form ax^2 +/- bx +/- c
            if (!deg.equals(2)) {
                stop(`Cannot complete square for degree ${deg.text()}`);
            }
            // Get the coeffs
            const coeffs = core.Algebra.coeffs(symbol, v);
            const a = coeffs[2];
            // Store the sign
            const sign = coeffs[1].sign();
            // Divide the linear term by two and square it
            const b = _.divide(coeffs[1], new NerdamerSymbol(2));
            // Add the difference to the constant
            const c = _.pow(b.clone(), new NerdamerSymbol(2));
            const sqrtA = math.sqrt(a);
            const e = _.divide(math.sqrt(c), sqrtA.clone());
            // Calculate d which is the constant
            const d = _.subtract(coeffs[0], _.pow(e.clone(), new NerdamerSymbol(2)));
            if (raw) {
                return [a, b, d];
            }
            // Compute the square part
            const sym = _.parse(br(`${sqrtA.clone()}*${v}${sign < 0 ? '-' : '+'}${e}`));
            return {
                a: sym,
                c: d,
                f: _.add(_.pow(sym.clone(), new NerdamerSymbol(2)), d.clone()),
            };
        },
        Simplify: {
            /**
             * @param {NerdamerSymbolType} symbol
             * @returns {[NerdamerSymbolType, NerdamerSymbolType, NerdamerSymbolType]}
             */
            strip(symbol) {
                const c = /** @type {NerdamerSymbolType} */ (_.parse(symbol.multiplier));
                symbol.toUnitMultiplier();
                const p = /** @type {NerdamerSymbolType} */ (_.parse(symbol.power));
                symbol.toLinear();
                return [c, p, symbol];
            },
            /**
             * @param {[NerdamerSymbolType, NerdamerSymbolType] | NerdamerSymbolType[]} cp
             * @param {NerdamerSymbolType} symbol
             * @returns {NerdamerSymbolType}
             */
            unstrip(cp, symbol) {
                const c = cp[0];
                const p = cp[1];
                const result = /** @type {NerdamerSymbolType} */ (_.multiply(c, _.pow(symbol, p)));
                return result;
            },
            /**
             * @param {NerdamerSymbolType} num
             * @param {NerdamerSymbolType} den
             * @returns {NerdamerSymbolType}
             */
            complexSimp(num, den) {
                const r1 = num.realpart();
                const i1 = num.imagpart();
                const r2 = den.realpart();
                const i2 = den.imagpart();
                // Apply complex arithmatic rule
                const ac = _.multiply(r1.clone(), r2.clone());
                const bd = _.multiply(i1.clone(), i2.clone());
                const bc = _.multiply(r2.clone(), i1);
                const ad = _.multiply(r1, i2.clone());
                const cd = _.add(_.pow(r2, new NerdamerSymbol(2)), _.pow(i2, new NerdamerSymbol(2)));

                return /** @type {NerdamerSymbolType} */ (
                    _.divide(_.add(_.add(ac, bd), _.multiply(_.subtract(bc, ad), NerdamerSymbol.imaginary())), cd)
                );
            },
            /**
             * Simplify trigonometric expressions.
             *
             * @param {NerdamerSymbolType} symbol
             * @returns {NerdamerSymbolType}
             */
            trigSimp(symbol) {
                let workDone = true;
                let iterations = 0;
                while (workDone && symbol.containsFunction(['cos', 'sin', 'tan'])) {
                    iterations++;
                    workDone = false;
                    symbol = symbol.clone();
                    // Remove power and multiplier
                    const symArray = __.Simplify.strip(symbol);
                    symbol = symArray.pop();
                    // The default return value is the symbol
                    let retval = symbol.clone();

                    // Rewrite the symbol
                    if (symbol.group === CP) {
                        let sym = new NerdamerSymbol(0);
                        symbol.each(x => {
                            // Rewrite the function
                            const tr = __.Simplify.trigSimp(x.fnTransform());
                            sym = /** @type {NerdamerSymbolType} */ (_.add(sym, tr));
                        }, true);

                        // Put back the power and multiplier and return
                        retval = /** @type {NerdamerSymbolType} */ (
                            _.pow(
                                _.multiply(new NerdamerSymbol(symbol.multiplier), sym),
                                new NerdamerSymbol(/** @type {FracType} */ (symbol.power))
                            )
                        );
                        workDone = retval.text() !== symbol.text();
                    } else if (symbol.group === CB) {
                        const n = symbol.getNum();
                        const d = symbol.getDenom();

                        // Try for tangent or fractions with tangent
                        if (
                            n.fname === 'sin' &&
                            d.fname === 'cos' &&
                            n.args[0].equals(d.args[0]) &&
                            /** @type {FracType} */ (n.power).equals(/** @type {FracType} */ (d.power))
                        ) {
                            retval = /** @type {NerdamerSymbolType} */ (
                                _.parse(
                                    core.Utils.format(
                                        '(({1})/({0}))*tan({2})^({3})',
                                        d.multiplier,
                                        n.multiplier,
                                        n.args[0],
                                        n.power
                                    )
                                )
                            );
                            workDone = true;
                        } else if (
                            n.fname === 'tan' &&
                            d.fname === 'sin' &&
                            n.args[0].equals(d.args[0]) &&
                            /** @type {FracType} */ (n.power).equals(/** @type {FracType} */ (d.power))
                        ) {
                            retval = /** @type {NerdamerSymbolType} */ (
                                _.parse(
                                    core.Utils.format(
                                        '(({1})/({0}))*cos({2})^(-({3}))',
                                        d.multiplier,
                                        n.multiplier,
                                        n.args[0],
                                        n.power
                                    )
                                )
                            );
                            workDone = true;
                        } else {
                            let t = new NerdamerSymbol(1);
                            const state = { workDone };
                            retval.each(x => {
                                if (x.fname === 'tan') {
                                    x = _.parse(
                                        core.Utils.format(
                                            '({0})*sin({1})^({2})/cos({1})^({2})',
                                            x.multiplier,
                                            __.Simplify._simplify(x.args[0]),
                                            x.power
                                        )
                                    );
                                    state.workDone = true;
                                } else if (x.containsFunction(['cos', 'sin', 'tan'])) {
                                    // Rewrite the function
                                    const y = __.Simplify.trigSimp(x);
                                    if (!x.equals(y)) {
                                        x = y;
                                        state.workDone = true;
                                    }
                                }
                                t = /** @type {NerdamerSymbolType} */ (_.multiply(t, x));
                            });
                            workDone = state.workDone;
                            retval = /** @type {NerdamerSymbolType} */ (t);
                        }
                    } else if ((symbol.fname === 'cos' || symbol.fname === 'sin') && symbol.args[0].group === CP) {
                        // Capture cos(x-pi/2) => sin(x) and sin(x+pi/2) = cos(x)
                        // but generalized
                        // test the sum for presence of a "n*pi/2" summands
                        let count = 0;
                        let newArg = new NerdamerSymbol(0);
                        const piOverTwo = _.parse('pi/2');
                        symbol.args[0].each(x => {
                            let c = /** @type {NerdamerSymbolType} */ (_.divide(x.clone(), piOverTwo.clone()));
                            c = __.Simplify._simplify(c);
                            c = core.Utils.evaluate(c);
                            if (isInt(c)) {
                                count += c.multiplier.num.toJSNumber();
                            } else {
                                newArg = /** @type {NerdamerSymbolType} */ (_.add(newArg, x));
                            }
                        });
                        if (count) {
                            count += symbol.fname === 'cos' ? 1 : 0;
                            count %= 4;
                            count += count < 0 ? 4 : 0;
                            // Console.log(count);
                            // debugger;
                            const results = ['sin({0})', 'cos({0})', '-sin({0})', '-cos({0})'];
                            const s = core.Utils.format(results[count], String(newArg));
                            retval = _.parse(s);
                            workDone = true;
                        } else if (Object.keys(symbol.args[0].symbols).length > 1) {
                            // Apply sin(a+-b) => sin(a)cos(b)+-cos(a)sin(b)
                            //   and cos(a+-b) => cos(a)cos(b)-+sin(a)sin(b)
                            const arg = symbol.args[0].clone();
                            const summands = Object.values(arg.symbols);
                            const a = summands[0];
                            const b = summands.slice(1);
                            const bStr = b.map(x => `(${x.text()})`).join('+');
                            let s;
                            if (symbol.fname === 'sin') {
                                s = core.Utils.format('sin({0})cos({1})+sin({1})cos({0})', a, bStr);
                            } else {
                                s = core.Utils.format('cos({0})cos({1})-sin({1})sin({0})', a, bStr);
                            }
                            retval = _.parse(s);
                            workDone = true;
                        }
                    } else if (
                        (symbol.fname === 'cos' || symbol.fname === 'sin') &&
                        symbol.args[0].multiplier.sign() === -1
                    ) {
                        // Sin(-x) => -sin(x), cos(-x) => cos(x)
                        // remove the minus from the argument
                        const newArg = symbol.args[0].clone().negate();
                        // Make the new trig call
                        let s = core.Utils.format(`${symbol.fname}({0})`, newArg);
                        if (symbol.fname === 'sin') {
                            s = `-${s}`;
                        }
                        retval = _.parse(s);
                        // Continue with the simpler form
                        workDone = true;
                    }
                    if (symbol.fname === 'sin' && symbol.args[0].multiplier.equals(2) && !symbol.args[0].equals(2)) {
                        // Sin(2x) => 2sin(x)cos(x)
                        // remove the minus from the argument
                        const newArg = symbol.args[0].clone().toUnitMultiplier();
                        // Make the new trig call
                        const s = core.Utils.format('2sin({0})cos({0})', newArg);
                        retval = _.parse(s);
                        // Continue with the simpler form
                        workDone = true;
                    }

                    retval = __.Simplify.unstrip(symArray, retval).distributeMultiplier();
                    symbol = retval;
                    // Safety check: prevent infinite loops
                    if (iterations > 10) {
                        break;
                    }
                }

                return symbol;
            },
            logArgSimp(fn, term) {
                // Console.log("----- log term: "+ term.text());
                // note: use symbol.equals
                if (term.value === '1' || term.value === String(1)) {
                    return new NerdamerSymbol(0);
                }
                // Work on all factors of the arg term
                // inintialize the sum
                let r = new NerdamerSymbol(0);
                // First up: the numerator's multiplier
                const m = term.multiplier.clone();
                // Console.log("----  multiplier: "+m);
                term.toUnitMultiplier();
                // Console.log("term with unit multiplier: "+term);

                if (!m.equals(1)) {
                    const a = core.Utils.format('({0}({1}))', fn, m);
                    // Console.log("m transformed: "+a);
                    r = /** @type {NerdamerSymbolType} */ (_.add(r, _.parse(a)));
                    // Console.log("m r: "+r.text());
                }
                // Now each factor, with its power
                // console.log("---- term factors");
                if (term.group === CB) {
                    // Product
                    term.each(x => {
                        x = x.clone();
                        const p = x.power.clone();
                        // Note: there will be no multiplier
                        // strip modifies the original
                        __.Simplify.strip(x);
                        // Console.log("factor: "+m+" * "+x+"^"+p+" = "+original);
                        const a = core.Utils.format('(({1})*{0}({2}))', fn, p, x);
                        // Console.log("factor transformed: "+a);
                        r = /** @type {NerdamerSymbolType} */ (_.add(r, _.parse(a)));
                        // Console.log("running sum: "+r.text());
                    });
                } else {
                    // Everything else
                    const x = term.clone();
                    const p = x.power.clone();
                    // Note: there will be no multiplier
                    // strip modifies the original
                    __.Simplify.strip(x);
                    // Console.log("factor: "+m+" * "+x+"^"+p+" = "+original);
                    const a = core.Utils.format('(({1})*{0}({2}))', fn, p, x);
                    // Console.log("factor transformed: "+r+"+"+a);
                    r = /** @type {NerdamerSymbolType} */ (_.add(r, _.parse(a)));
                    // Console.log("running sum: "+r.text());
                }
                // Console.log("result: "+r.text());
                return r;
            },
            logSimp(symbol) {
                if (symbol.group === FN && (symbol.fname === 'log' || symbol.fname === 'log10')) {
                    // Console.log();
                    // console.log("Initial: "+symbol.text());
                    // remove power and multiplier
                    const _original = symbol.clone();
                    const symArray = __.Simplify.strip(symbol);
                    symbol = symArray.pop();

                    // Work on the argument
                    const arg = symbol.args[0].clone();
                    const n = arg.getNum().clone();
                    // Console.log("n: "+n.text());
                    const d = arg.getDenom().clone();
                    // Console.log("d: "+d.text());
                    const fn = symbol.fname;

                    let retval = __.Simplify.logArgSimp(fn, n);
                    if (!d.equals(1)) {
                        const rd = __.Simplify.logArgSimp(fn, d);
                        retval = /** @type {NerdamerSymbolType} */ (_.subtract(retval, rd));
                    }

                    retval = __.Simplify.unstrip(symArray, retval).distributeMultiplier();
                    symbol = retval;
                    // Console.log("result: "+symbol.text());
                } else if (symbol.containsFunction(['log', 'log10'])) {
                    for (const termkey in symbol.symbols) {
                        if (!Object.hasOwn(symbol.symbols, termkey)) {
                            continue;
                        }
                        const term = symbol.symbols[termkey];
                        symbol.symbols[termkey] = __.Simplify.logSimp(term);
                    }
                }

                return symbol;
            },
            /**
             * Compresses sqrt expressions in fractions.
             *
             * @param {NerdamerSymbolType} symbol The symbol
             * @param {NerdamerSymbolType} num Numerator
             * @param {NerdamerSymbolType} den Denominator
             * @returns {NerdamerSymbolType}
             */
            _sqrtCompression(symbol, num, den) {
                // Return symbol;
                // preserve power and multiplier
                const symArray = __.Simplify.strip(symbol);

                // Helper functions
                const isABS = s => s.fname === 'abs';
                const getArg = s => s.args[0];
                const absArg = s => (isABS(s) ? getArg(s) : null);
                const isUnit = s => s.type === S && s.value.startsWith('baseunit_');

                // Main workhorse function
                const cancel = (a, sqrt) => {
                    const sqrtArg = getArg(sqrt);
                    // Abs(x):sqrt(x) => sqrt(x)
                    if (sqrtArg.equals(absArg(a))) {
                        return [sqrt, null];
                    }
                    // Unit(x):sqrt(x) => sqrt(x)
                    if (sqrtArg.equals(a) && isUnit(a)) {
                        return [sqrt, null];
                    }

                    // N*sqrt(a):d*sqrt(x) => (n/d)*sqrt(a/x)
                    // if (a.isSQRT()) {
                    //     let newArg = getArg(a);
                    //     let m = new NerdamerSymbol(a.multiplier);
                    //     m = _.divide(m, sqrt.multiplier);
                    //     newArg = _.divide(newArg, sqrtArg);
                    //     const combinedSqrt = core.Utils.format('sqrt({0})', newArg);
                    //     const result = _.multiply(new NerdamerSymbol(m), _.parse(combinedSqrt));
                    //     return [result, null];
                    // }

                    // nothing to be done
                    return [null, sqrt];
                };

                let workDone;
                let totalWorkDone = false;

                const cancelTerms = (top, bottom) => {
                    for (let i = 0; i < top.length; i++) {
                        // Examine the first top symbol
                        let sqrt = top[i];
                        if (!sqrt.isSQRT()) {
                            continue;
                        }
                        // It's a sqrt. try to cancel it against each
                        // bottom term
                        for (let j = 0; j < bottom.length; j++) {
                            let term = bottom[j];
                            [term, sqrt] = cancel(term, sqrt);
                            if (term !== null) {
                                // We found a match, substitute the remains and exit here
                                bottom[j] = term;
                                workDone = true;
                                totalWorkDone = true;
                                break;
                            }
                        }
                        // Whatever remains of sqrt gets put back
                        top[i] = sqrt;
                        top = top.filter(x => x);
                        bottom = bottom.filter(x => x);
                    }
                    return [top, bottom];
                };

                // Look for sqrt terms in products in num and den
                // if we find any, combine them with other terms

                // first, collect all factors in numerator and denominator
                let numSymbols = num.collectFactors();
                let denSymbols = den.collectFactors();

                // Now cancel terms until nothing to cancel was found
                do {
                    workDone = false;
                    [numSymbols, denSymbols] = cancelTerms(numSymbols, denSymbols);
                    [denSymbols, numSymbols] = cancelTerms(denSymbols, numSymbols);
                } while (workDone);

                if (totalWorkDone) {
                    // Reassemble the fraction symbol
                    symbol = /** @type {NerdamerSymbolType} */ (
                        numSymbols.reduce(
                            (acc, s) => (acc = /** @type {NerdamerSymbolType} */ (_.multiply(acc, s))),
                            new NerdamerSymbol(1)
                        )
                    );
                    symbol = /** @type {NerdamerSymbolType} */ (
                        denSymbols.reduce(
                            (acc, s) => (acc = /** @type {NerdamerSymbolType} */ (_.divide(acc, s))),
                            symbol
                        )
                    );
                }

                // Add power etc. back in
                symbol = __.Simplify.unstrip(symArray, symbol);

                return symbol;
            },

            fracSimp(symbol) {
                // Try a quick simplify of imaginary numbers
                let den = symbol.getDenom();
                let num = symbol.getNum();

                if (num.isImaginary() && den.isImaginary()) {
                    symbol = __.Simplify.complexSimp(num, den);
                }

                if (symbol.isComposite()) {
                    if (/** @type {FracType} */ (symbol.power).gt(1)) {
                        symbol = /** @type {NerdamerSymbolType} */ (_.expand(symbol));
                    }

                    const symbols = symbol.collectSymbols();
                    // Assumption 1.
                    // since it's a composite, it has a length of at least 1
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let retval;
                    /** @type {NerdamerSymbolType} */
                    let a;
                    /** @type {NerdamerSymbolType} */
                    let b;
                    /** @type {NerdamerSymbolType} */
                    let d1;
                    /** @type {NerdamerSymbolType} */
                    let d2;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let n1;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let n2;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let s;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let x;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let y;
                    /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                    let c;
                    a = /** @type {NerdamerSymbolType} */ (symbols.pop()); // Grab the first symbol
                    // loop through each term and make denominator common
                    while (symbols.length) {
                        b = /** @type {NerdamerSymbolType} */ (symbols.pop()); // Grab the second symbol
                        d1 = /** @type {NerdamerSymbolType} */ (_.parse(a.getDenom()));
                        d2 = /** @type {NerdamerSymbolType} */ (_.parse(b.getDenom()));
                        n1 = a.getNum();
                        n2 = b.getNum();
                        c = _.multiply(d1.clone(), d2.clone());
                        x = _.multiply(n1, d2);
                        y = _.multiply(n2, d1);
                        s = _.add(x, y);
                        a = /** @type {NerdamerSymbolType} */ (_.divide(s, c));
                    }
                    den = /** @type {NerdamerSymbolType} */ (_.expand(a.getDenom()));
                    num = /** @type {NerdamerSymbolType} */ (_.expand(a.getNum()));
                    // Simplify imaginary
                    if (num.isImaginary() && den.isImaginary()) {
                        retval = __.Simplify.complexSimp(num, den);
                    } else {
                        retval = _.divide(num, den);
                    }

                    // We've already hit the simplest form so return that
                    if (/** @type {NerdamerSymbolType} */ (retval).equals(symbol)) {
                        return symbol;
                    }

                    // Otherwise simplify it some more
                    return __.Simplify._simplify(retval);
                }
                symbol = __.Simplify._sqrtCompression(
                    symbol,
                    /** @type {NerdamerSymbolType} */ (num),
                    /** @type {NerdamerSymbolType} */ (den)
                );
                symbol = /** @type {NerdamerSymbolType} */ (__.Simplify.simpleFracSimp(symbol));
                return symbol;
            },
            simpleFracSimp(symbol) {
                let den = /** @type {NerdamerSymbolType} */ (symbol.getDenom());
                let num = /** @type {NerdamerSymbolType} */ (symbol.getNum());
                /** @type {NerdamerSymbolType | VectorType | MatrixType} */
                let retval;
                den = /** @type {NerdamerSymbolType} */ (_.expand(den));
                num = /** @type {NerdamerSymbolType} */ (_.expand(num));
                // Simplify imaginary
                if (num.isImaginary() && den.isImaginary()) {
                    retval = __.Simplify.complexSimp(num, den);
                } else {
                    retval = _.divide(num, den);
                }
                // We've already hit the simplest form so return that
                if (/** @type {NerdamerSymbolType} */ (retval).equals(symbol)) {
                    return symbol;
                }
                // Otherwise simplify it some more
                // retval = __.Simplify._simplify(retval);
                return retval;
            },
            ratSimp(symbol) {
                if (symbol.group === CB) {
                    const den = symbol.getDenom();
                    const num = symbol.getNum().distributeMultiplier();
                    const d = __.Simplify.fracSimp(den);
                    const n = __.Simplify.fracSimp(num);
                    symbol = /** @type {NerdamerSymbolType} */ (_.divide(n, d));
                }
                return symbol;
            },
            sqrtSimp(symbol, _sym_array) {
                let retval;
                let workDone = false;

                const original = symbol.clone();
                try {
                    // Debuglevel(1);
                    // debugout("input:  "+symbol.toString());

                    if (symbol.isSQRT()) {
                        // Symbol is itself sqrt
                        // save outer multiplier
                        const mOuter = symbol.multiplier.clone();

                        // Now factor it
                        const sqrtArg = symbol.args[0].clone();
                        const factored = __.Factor.factorInner(sqrtArg);

                        // Get a sanitized version of the argument's multiplier
                        const m = _.parse(factored.multiplier);
                        // And its sign
                        const sign = m.sign();

                        // Make an initial return value
                        retval = new NerdamerSymbol(1);
                        let arg;

                        if (factored.group === CB) {
                            // Monomial arg
                            let rem = new NerdamerSymbol(1);

                            factored.each(x => {
                                x = _.parse(x);
                                if (x.group === N) {
                                    const trial = _.sqrt(x.clone());

                                    // Multiply back sqrt if it's an integer otherwise just put back the number
                                    if (isInt(trial)) {
                                        retval = /** @type {NerdamerSymbolType} */ (_.multiply(retval, trial));
                                    } else {
                                        rem = /** @type {NerdamerSymbolType} */ (_.multiply(rem, x));
                                    }
                                } else {
                                    rem = /** @type {NerdamerSymbolType} */ (_.multiply(rem, x));
                                }
                            });
                            const t = /** @type {NerdamerSymbolType} */ (_.multiply(rem, _.parse(sign)));
                            arg = /** @type {NerdamerSymbolType} */ (_.sqrt(t.clone()));

                            // Expand if it's imaginary
                            if (arg.isImaginary()) {
                                arg = /** @type {NerdamerSymbolType} */ (
                                    _.sqrt(/** @type {NerdamerSymbolType} */ (_.expand(t.clone())))
                                );
                            }
                        } else {
                            // Put together the argument with the sign
                            // but without the multiplier
                            arg = factored.clone().toUnitMultiplier();
                            arg = /** @type {NerdamerSymbolType} */ (_.multiply(arg, new NerdamerSymbol(sign)));
                            arg = /** @type {NerdamerSymbolType} */ (_.sqrt(arg));
                        }

                        // Put the result back
                        retval = _.multiply(retval, arg);
                        // Put back the multiplier
                        retval = _.pow(retval, _.parse(symbol.power));
                        retval = _.multiply(retval, _.sqrt(m.abs()));
                        retval = _.multiply(retval, _.parse(mOuter));
                        workDone = true;
                    } else if (symbol.isComposite() && symbol.isLinear()) {
                        // Polynomial or CP => sum of things
                        retval = new NerdamerSymbol(0);
                        symbol.each(x => {
                            retval = _.add(retval, __.Simplify.sqrtSimp(x));
                        }, true);
                        // Put back the multiplier and power
                        retval = _.pow(retval, _.parse(symbol.power));
                        retval = _.multiply(retval, _.parse(symbol.multiplier));
                        workDone = true;
                    } else if (symbol.group === CB) {
                        // Monomial
                        retval = new NerdamerSymbol(1);
                        symbol.each(x => {
                            const simp = __.Simplify.sqrtSimp(x);
                            retval = _.multiply(retval, simp);
                        });
                        // Put back the power and multiplier
                        retval = _.pow(retval, _.parse(symbol.power));
                        retval = _.multiply(retval, _.parse(symbol.multiplier));
                        workDone = true;
                    }

                    if (!workDone) {
                        if (retval && !isInt(retval)) {
                            // If we can't even pull an integer out, revert
                            // to the cautious fallback
                            retval = null;
                        }
                    }

                    // Fallback: original symbol
                    retval ||= _.parse(symbol);
                    // Debugout("result: "+retval.toString());
                    // debugout("");
                    return retval;
                } catch (error) {
                    if (error.message === 'timeout') {
                        throw error;
                    }
                    // Error in sqrtsimp - return original symbol
                    return original;
                } finally {
                    // Debuglevel(-1);
                }
            },
            /**
             * Unused. The goal is to substitute out patterns but it currently doesn't work.
             *
             * @param {NerdamerSymbolType} symbol
             * @returns {Array} The symbol and the matched patterns
             */
            patternSub(symbol) {
                const patterns = {};

                const hasCP = function (sym) {
                    let found = false;
                    sym.each(x => {
                        if (x.group === CP) {
                            found = true;
                        } else if (x.symbols) {
                            found = hasCP(x);
                        }
                    });

                    return found;
                };

                const collect = function (sym) {
                    // We loop through each symbol looking for anything in the simplest
                    // form of ax+byz+...
                    sym.each(x => {
                        // Items of group N,P,S, need to apply
                        if (!x.symbols && x.group !== FN) {
                            return;
                        }

                        // Check to see if it has any symbols of group CP
                        // Get the patterns in that symbol instead if it has anything of group CP
                        if (hasCP(x)) {
                            collect(x);
                        } else if (!patterns[x.value]) {
                            const u = core.Utils.getU(symbol);
                            // Get a u value and mark it for subsitution
                            patterns[x.value] = u;
                            symbol = symbol.sub(x.value, u);
                        }
                    }, true);
                };

                // Collect a list of patterns
                collect(symbol);

                return [symbol, patterns];
            },
            simplify(symbol) {
                if (symbol.simplify) {
                    return symbol.simplify();
                }
                let retval = __.Simplify._simplify(symbol);
                retval = retval.pushMinus();
                retval = _.parse(retval);
                return retval;
            },
            _simplify(symbol) {
                // Debuglevel(1);
                // debugout("input to _simplify: "+symbol.text());
                // try {
                // remove the multiplier to make calculation easier;
                const symArray = __.Simplify.strip(/** @type {NerdamerSymbolType} */ (symbol).clone());
                symbol = /** @type {NerdamerSymbolType | VectorType | MatrixType} */ (symArray.pop());
                // Remove gcd from denominator
                symbol = __.Simplify.fracSimp(/** @type {NerdamerSymbolType} */ (symbol));
                // Nothing more to do
                if (
                    /** @type {NerdamerSymbolType} */ (symbol).isConstant() ||
                    /** @type {NerdamerSymbolType} */ (symbol).group === core.groups.S
                ) {
                    symArray.push(/** @type {NerdamerSymbolType} */ (symbol));
                    const ret = __.Simplify.unstrip(symArray, /** @type {NerdamerSymbolType} */ (symbol));
                    // Debugout("final result: "+ret.text());
                    return ret;
                }
                // Console.log("array: "+symArray);

                // let patterns;

                let simplified = /** @type {NerdamerSymbolType} */ (symbol).clone(); // Make a copy

                // [simplified, patterns] = __.Simplify.patternSub(symbol);

                // Simplify sqrt within the symbol
                // todo: why does this break calculus tests?
                simplified = /** @type {NerdamerSymbolType} */ (__.Simplify.sqrtSimp(simplified, symArray));

                // Try trig simplificatons e.g. cos(x)^2+sin(x)^2
                simplified = __.Simplify.trigSimp(simplified);

                // Try log simplificatons e.g. log(a/b)=> log(a)-log(b)
                simplified = __.Simplify.logSimp(simplified);

                // Simplify common denominators
                simplified = /** @type {NerdamerSymbolType} */ (__.Simplify.ratSimp(simplified));

                // First go for the "cheapest" simplification which may eliminate
                // your problems right away. factor -> evaluate. Remember
                // that there's no need to expand since factor already does that

                // console.log("before factor: "+simplified.text());
                simplified = __.Factor.factorInner(simplified);
                // Console.log("after factor: "+simplified.text());

                // If the simplified is a sum then we can make a few more simplifications
                // e.g. simplify(1/(x-1)+1/(1-x)) as per issue #431
                // console.log("before sums: "+simplified.text());
                if (simplified.group === core.groups.CP && simplified.isLinear()) {
                    const m = simplified.multiplier.clone();
                    simplified.toUnitMultiplier(); // Strip the multiplier
                    let r = new NerdamerSymbol(0);
                    // Return the sum of simplifications
                    simplified.each(x => {
                        const s = __.Simplify._simplify(x);
                        r = /** @type {NerdamerSymbolType} */ (_.add(r, s));
                    });
                    simplified = r;
                    // Mult on back the multiplier we saved here
                    simplified = /** @type {NerdamerSymbolType} */ (_.multiply(simplified, new NerdamerSymbol(m)));
                    if (simplified.multiplier.equals(-1)) {
                        simplified.distributeMultiplier();
                    }
                    // Place back original multiplier and return
                    simplified = __.Simplify.unstrip(symArray, simplified);
                    // Debugout("final result: "+simplified.text());
                    return simplified;
                }

                // Place back original multiplier and return
                simplified = __.Simplify.unstrip(symArray, simplified);
                // Debugout("final result: "+simplified.text());
                return simplified;
                // } finally {
                //     // debuglevel(-1);
                // }
            },
        },

        Classes: {
            Polynomial,
            Factors: /** @type {FactorsConstructor} */ (/** @type {unknown} */ (Factors)),
            MVTerm,
        },
    });

    // Add a link to simplify
    core.Expression.prototype.simplify = function simplify() {
        core.Utils.armTimeout();
        try {
            let retval;
            // Equation?
            if (typeof this.symbol.LHS === 'undefined') {
                retval = new core.Expression(__.Simplify.simplify(this.symbol));
            } else {
                // Don't have access to equation here, so we clone instead
                const eq = this.symbol.clone();
                eq.LHS = __.Simplify.simplify(eq.LHS);
                eq.RHS = __.Simplify.simplify(eq.RHS);
                retval = eq;
            }
            return retval;
        } catch (error) {
            if (error.message === 'timeout') {
                throw error;
            }
            return this;
        } finally {
            core.Utils.disarmTimeout();
        }
    };

    core.Collection.prototype.simplify = function simplify() {
        this.elements = this.elements.map(e => __.Simplify.simplify(e));
        return this;
    };

    core.Matrix.prototype.simplify = function simplify() {
        this.elements = this.elements.map(row => row.map(e => __.Simplify.simplify(e)));
        return this;
    };

    nerdamer.useAlgebraDiv = function useAlgebraDiv() {
        const _originalDivide = (__.divideFn = _.divide);
        let calls = 0; // Keep track of how many calls were made
        _.divide = function divide(a, b) {
            calls++;
            let ans;
            if (calls === 1) // Check if this is the first call. If it is use algebra divide
            {
                ans = core.Algebra.divide(/** @type {NerdamerSymbolType} */ (a), /** @type {NerdamerSymbolType} */ (b));
            } // Otherwise use parser divide
            else {
                ans = divide(a, b);
            }
            calls = 0; // Reset the number of calls back to none
            return ans;
        };
    };

    nerdamer.useParserDiv = function useParserDiv() {
        if (__.divideFn) {
            _.divide = __.divideFn;
        }
        delete __.divideFn;
    };

    nerdamer.register([
        {
            name: 'factor',
            visible: true,
            numargs: 1,
            build() {
                return __.Factor.factor;
            },
        },
        {
            name: 'simplify',
            visible: true,
            numargs: 1,
            build() {
                return __.Simplify.simplify;
            },
        },
        {
            name: 'gcd',
            visible: true,
            numargs: [1],
            build() {
                return __.gcd;
            },
        },
        {
            name: 'lcm',
            visible: true,
            numargs: [1],
            build() {
                return __.lcm;
            },
        },
        {
            name: 'roots',
            visible: true,
            numargs: -1,
            build() {
                return __.roots;
            },
        },
        {
            name: 'divide',
            visible: true,
            numargs: 2,
            build() {
                return __.divide;
            },
        },
        {
            name: 'div',
            visible: true,
            numargs: 2,
            build() {
                return __.div;
            },
        },
        {
            name: 'partfrac',
            visible: true,
            numargs: [1, 2],
            build() {
                return __.PartFrac.partfrac;
            },
        },
        {
            name: 'deg',
            visible: true,
            numargs: [1, 2],
            build() {
                return __.degree;
            },
        },
        {
            name: 'coeffs',
            visible: true,
            numargs: [1, 2],
            build() {
                const f = function (...args) {
                    const coeffs = __.coeffs(/** @type {NerdamerSymbolType} */ (args[0]), args[1]);
                    return new core.Vector(coeffs);
                };
                return f;
            },
        },
    ]);

    // Register coeffs with direct access to nerdamer that preserves symbolic constants
    // The standard updateAPI wrapper uses PARSE2NUMBER which converts pi, e, sqrt(2) to rationals
    // This version parses arguments without PARSE2NUMBER to preserve symbolic constants
    /** @type {any} */ (nerdamer).coeffs = function coeffs(...args) {
        const parser = core.PARSER;
        // Parse arguments WITHOUT PARSE2NUMBER to preserve symbolic constants like pi, e, sqrt(2)
        for (let i = 0; i < args.length; i++) {
            if (typeof args[i] === 'string') {
                args[i] = parser.parse(/** @type {string} */ (args[i]));
            } else if (args[i] && /** @type {ExpressionType} */ (args[i]).symbol) {
                // It's an Expression, get the symbol
                args[i] = /** @type {ExpressionType} */ (args[i]).symbol.clone();
            } else if (core.Utils.isSymbol(args[i])) {
                args[i] = /** @type {NerdamerSymbolType} */ (args[i]).clone();
            }
        }
        const resultCoeffs = __.coeffs(/** @type {NerdamerSymbolType} */ (args[0]), args[1]);
        return new core.Expression(/** @type {VectorType} */ (new core.Vector(resultCoeffs)));
    };

    nerdamer.register([
        {
            name: 'line',
            visible: true,
            numargs: [2, 3],
            build() {
                return __.line;
            },
        },
        {
            name: 'sqcomp',
            visible: true,
            numargs: [1, 2],
            build() {
                const f = function (x, v) {
                    try {
                        v ||= variables(x)[0];
                        const sq = __.sqComplete(x.clone(), v);
                        return /** @type {{ f: NerdamerSymbol; a: NerdamerSymbol; c: NerdamerSymbol }} */ (sq).f;
                    } catch (e) {
                        if (e.message === 'timeout') {
                            throw e;
                        }
                        return x;
                    }
                };
                return f;
            },
        },
    ]);
    nerdamer.updateAPI();
})();
