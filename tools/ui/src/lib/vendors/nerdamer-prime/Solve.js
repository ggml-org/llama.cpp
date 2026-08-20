/*
 * Author : Martin Donk
 * Website : http://www.nerdamer.com
 * Email : martin.r.donk@gmail.com
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
 *   Constructor types (for factory functions)
 *
 * @typedef {import('./index').NerdamerCore.SymbolConstructor} SymbolConstructor
 *
 * @typedef {import('./index').NerdamerCore.VectorConstructor} VectorConstructor
 *
 *   Module types
 *
 * @typedef {import('./index').NerdamerCore.AlgebraModule} AlgebraModuleType
 *
 * @typedef {import('./index').NerdamerCore.CalculusModule} CalculusModuleType
 *
 * @typedef {import('./index').NerdamerCore.FactorSubModule} FactorSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.SimplifySubModule} SimplifySubModuleType
 *
 * @typedef {import('./index').NerdamerCore.IntegrationSubModule} IntegrationSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.AlgebraClassesSubModule} AlgebraClassesSubModuleType
 *
 * @typedef {import('./index').NerdamerCore.DecomposeResultObject} DecomposeResultType
 *
 * @typedef {import('./index').NerdamerCore.SolveModule} SolveModuleType
 *
 *   Utility types
 *
 * @typedef {import('./index').NerdamerCore.Utils} UtilsInterface
 *
 * @typedef {import('./index').NerdamerCore.Build} BuildInterface
 *
 * @typedef {import('big-integer').BigInteger} BigIntegerType
 *
 *   Equation instance type
 *
 * @typedef {import('./index').NerdamerCore.EquationInstance} EquationInstanceType
 *
 *   Solution result types
 *
 * @typedef {import('./index').NerdamerCore.SystemSolutionResult} SystemSolutionResultType
 *
 * @typedef {import('./index').NerdamerCore.SystemSolutionValue} SystemSolutionValueType
 *
 * @typedef {import('./index').NerdamerCore.CircleSolutionResult} CircleSolutionResultType
 *
 * @typedef {(NerdamerSymbolType | EquationInstanceType | string)[]} SolveEquationArray
 */

// Check if nerdamer exists globally (browser) or needs to be required (Node.js)
let nerdamer = typeof globalThis !== 'undefined' && globalThis.nerdamer ? globalThis.nerdamer : undefined;
if (typeof module !== 'undefined' && nerdamer === undefined) {
    nerdamer = require('./nerdamer.core.js');
    require('./Calculus.js');
    require('./Algebra.js');
}

/** @returns {SolveModuleType} */
(function initSolveModule() {
    // Handle imports
    const core = nerdamer.getCore();
    const _ = core.PARSER;
    /** @type {AlgebraModuleType} */
    const _A = /** @type {AlgebraModuleType} */ (core.Algebra);
    /** @type {CalculusModuleType} */
    const _C = /** @type {CalculusModuleType} */ (core.Calculus);
    const { integration } = /** @type {{ integration: IntegrationSubModuleType }} */ (_C);
    const { decompose_arg: explode } = integration;
    const {
        Factor,
        Simplify,
        Classes: AlgebraClasses,
    } = /** @type {{ Factor: FactorSubModuleType; Simplify: SimplifySubModuleType; Classes: AlgebraClassesSubModuleType }} */ (
        _A
    );
    const { evaluate, remove, format, knownVariable, isSymbol, variables, range } = core.Utils;
    const { build } = core.Build;
    const { NerdamerSymbol } = core;
    const { S, PL, CB, CP, FN } = core.groups;
    const { Settings } = core;
    const { isArray } = core.Utils;

    // The search radius for the roots
    core.Settings.SOLVE_RADIUS = 1000;
    // The maximum number to fish for on each side of the zero
    core.Settings.ROOTS_PER_SIDE = 10;
    // Covert the number to multiples of pi if possible
    core.Settings.make_pi_conversions = false;
    // The step size
    core.Settings.STEP_SIZE = 0.1;

    // The epsilon size
    core.Settings.EPSILON = 2e-13;
    // The maximum iterations for Newton's method
    core.Settings.MAX_NEWTON_ITERATIONS = 200;
    // The epsilon used in Newton's iteration
    // core.Settings.NEWTON_EPSILON = Number.EPSILON * 2;
    core.Settings.NEWTON_EPSILON = 2e-15;

    // The maximum number of time non-linear solve tries another jump point
    core.Settings.MAX_NON_LINEAR_TRIES = 12;
    // The amount of iterations the function will start to jump at
    core.Settings.NON_LINEAR_JUMP_AT = 50;
    // The size of the jump
    core.Settings.NON_LINEAR_JUMP_SIZE = 100;
    // The original starting point for nonlinear solving
    core.Settings.NON_LINEAR_START = 0.01;
    // When points are generated as starting points for Newton's method, they are sliced into small
    // slices to make sure that we have convergence on the right point. This defines the
    // size of the slice
    core.Settings.NEWTON_SLICES = 200;
    // The distance in which two solutions are deemed the same
    core.Settings.SOLUTION_PROXIMITY = 1e-14;
    // Indicate wheter to filter the solutions are not
    core.Settings.FILTER_SOLUTIONS = true;
    // The maximum number of recursive calls
    core.Settings.MAX_SOLVE_DEPTH = 10;
    // The tolerance that's considered close enough to zero
    core.Settings.ZERO_EPSILON = 1e-9;
    // The maximum iteration for the bisection method incase of some JS strangeness
    core.Settings.MAX_BISECTION_ITER = 2000;
    // The tolerance for the bisection method
    core.Settings.BI_SECTION_EPSILON = 1e-12;

    core.NerdamerSymbol.prototype.hasTrig = function hasTrig() {
        return this.containsFunction(['cos', 'sin', 'tan', 'cot', 'csc', 'sec']);
    };

    core.NerdamerSymbol.prototype.hasNegativeTerms = function hasNegativeTerms() {
        if (this.isComposite()) {
            for (const x in this.symbols) {
                if (!Object.hasOwn(this.symbols, x)) {
                    continue;
                }
                const sym = this.symbols[x];
                if ((sym.group === PL && sym.hasNegativeTerms()) || this.symbols[x].power.lessThan(0)) {
                    return true;
                }
            }
        }
        return false;
    };

    /* Nerdamer version 0.7.x and up allows us to make better use of operator overloading
     * As such we can have this data type be supported completely outside of the core.
     * This is an equation that has a left hand side and a right hand side
     */
    /**
     * Equation class representing LHS = RHS.
     *
     * @class
     * @param {NerdamerSymbolType} lhs - The left hand side symbol
     * @param {NerdamerSymbolType} rhs - The right hand side symbol
     */
    function Equation(lhs, rhs) {
        if (
            (rhs.isConstant() && lhs.isConstant() && !lhs.equals(rhs)) ||
            (lhs.equals(core.Settings.IMAGINARY) && rhs.isConstant(true)) ||
            (rhs.equals(core.Settings.IMAGINARY) && lhs.isConstant(true))
        ) {
            throw new core.exceptions.NerdamerValueError(`${lhs.toString()} does not equal ${rhs.toString()}`);
        }
        /** @type {NerdamerSymbolType} */
        this.LHS = lhs; // Left hand side
        /** @type {NerdamerSymbolType} */
        this.RHS = rhs; // Right and side
    }
    // UTILS ##!!

    Equation.prototype = {
        toString() {
            return `${this.LHS.toString()}=${this.RHS.toString()}`;
        },
        text(option) {
            return `${this.LHS.text(option)}=${this.RHS.text(option)}`;
        },
        /**
         * Brings the equation to LHS (sets RHS to zero).
         *
         * @param {boolean} [expand] - Whether to expand the result
         * @returns {NerdamerSymbolType} The LHS with RHS subtracted
         */
        toLHS(expand) {
            expand = !!expand;
            const eqn = this.removeDenom();
            let a = eqn.LHS;
            let b = eqn.RHS;

            if (a.isConstant(true) && !b.isConstant(true)) {
                // Swap them to avoid confusing parser and cause an infinite loop
                [a, b] = [b, a];
            }
            const _t = /** @type {NerdamerSymbolType} */ (_.subtract(a, b));
            /** @type {NerdamerSymbolType} */
            let retval = expand ? /** @type {NerdamerSymbolType} */ (_.expand(_t)) : _t;

            // Quick workaround for issue #636
            // This basically borrows the removeDenom method from the Equation class.
            // TODO: Make this function a stand-alone function
            retval = new Equation(retval, new NerdamerSymbol(0)).removeDenom().LHS;

            return retval;
        },
        /**
         * Removes denominators from both sides.
         *
         * @returns {Equation} Equation with denominators removed
         */
        removeDenom() {
            let a = this.LHS.clone();
            let b = this.RHS.clone();
            // Remove the denominator on both sides
            const den = /** @type {NerdamerSymbolType} */ (_.multiply(a.getDenom(), b.getDenom()));
            a = /** @type {NerdamerSymbolType} */ (_.expand(_.multiply(a, den.clone())));
            b = /** @type {NerdamerSymbolType} */ (_.expand(_.multiply(b, den)));
            // Swap the groups
            if (b.group === CP && b.group !== CP) {
                const t = a;
                a = b;
                b = t; // Swap
            }

            // Scan to eliminate denominators
            if (a.group === CB) {
                let t = new NerdamerSymbol(a.multiplier);
                /** @type {NerdamerSymbolType} */
                let newRHS = b.clone();
                a.each(y => {
                    if (y.power.lessThan(0)) {
                        newRHS = /** @type {NerdamerSymbolType} */ (_.divide(newRHS, y));
                    } else {
                        t = /** @type {NerdamerSymbolType} */ (_.multiply(t, y));
                    }
                });
                a = t;
                b = newRHS;
            } else if (a.group === CP) {
                // The logic: loop through each and if it has a denominator then multiply it out on both ends
                // and then start over
                for (const x in a.symbols) {
                    if (!Object.hasOwn(a.symbols, x)) {
                        continue;
                    }
                    const sym = a.symbols[x];
                    if (sym.group === CB) {
                        for (const y in sym.symbols) {
                            if (!Object.hasOwn(sym.symbols, y)) {
                                continue;
                            }
                            const sym2 = sym.symbols[y];
                            if (sym2.power.lessThan(0)) {
                                const result = new Equation(
                                    /** @type {NerdamerSymbolType} */ (
                                        _.expand(_.multiply(sym2.clone().toLinear(), a))
                                    ),
                                    /** @type {NerdamerSymbolType} */ (_.expand(_.multiply(sym2.clone().toLinear(), b)))
                                );
                                return result;
                            }
                        }
                    }
                }
            }

            return new Equation(a, b);
        },
        /**
         * Creates a copy of this equation.
         *
         * @returns {Equation}
         */
        clone() {
            return new Equation(this.LHS.clone(), this.RHS.clone());
        },
        /**
         * Substitutes a value for a variable on both sides.
         *
         * @param {NerdamerSymbolType} x - Variable to replace
         * @param {NerdamerSymbolType} y - Value to substitute
         * @returns {Equation}
         */
        sub(x, y) {
            const clone = this.clone();
            clone.LHS = clone.LHS.sub(x.clone(), y.clone());
            clone.RHS = clone.RHS.sub(x.clone(), y.clone());
            return clone;
        },
        /**
         * Checks if the equation evaluates to zero.
         *
         * @returns {boolean}
         */
        isZero() {
            return core.Utils.evaluate(this.toLHS()).equals(0);
        },
        /**
         * Returns LaTeX representation.
         *
         * @param {string} [option]
         * @returns {string}
         */
        latex(option) {
            return [this.LHS.latex(option), this.RHS.latex(option)].join('=');
        },
    };
    // Overwrite the equals function
    /**
     * Creates an Equation from two symbols. This extends the parser's equals function to return Equation objects.
     *
     * @param {NerdamerSymbolType} a
     * @param {NerdamerSymbolType} b
     * @returns {Equation}
     */
    // @ts-ignore - Overriding parser.equals to return Equation instead of Symbol
    _.equals = function equals(a, b) {
        return new Equation(a, b);
    };

    // Extend simplify
    (function extendSimplifyForEquations() {
        const simplify = _.functions.simplify[0];
        _.functions.simplify[0] = function simplifyWithEquationSupport(symbol) {
            if (symbol instanceof Equation) {
                symbol.LHS = simplify(symbol.LHS);
                symbol.RHS = simplify(symbol.RHS);
                return symbol;
            }
            // Just call the original simplify
            return simplify(symbol);
        };
    })();

    /**
     * Sets two expressions equal
     *
     * @param {NerdamerSymbolType} symbol
     * @returns {Equation}
     */
    core.Expression.prototype.equals = function equals(symbol) {
        if (symbol instanceof core.Expression) {
            symbol = symbol.symbol;
        } // Grab the symbol if it's an expression
        const eq = new Equation(this.symbol, symbol);
        return eq;
    };

    core.Expression.prototype.solveFor = function solveFor(x) {
        core.Utils.armTimeout();
        try {
            const { symbol } = this;
            if (this.symbol instanceof Equation) {
                // Exit right away if we already have the answer
                // check the LHS
                if (this.symbol.LHS.isConstant() && this.symbol.RHS.equals(x)) {
                    return [new core.Expression(this.symbol.LHS)];
                }

                // Check the RHS
                if (this.symbol.RHS.isConstant() && this.symbol.LHS.equals(x)) {
                    return [new core.Expression(this.symbol.RHS)];
                }
            }

            const terms = solve(symbol, x);
            const result = terms.map(term => {
                term = /** @type {NerdamerSymbolType} */ (
                    Simplify.simplify(/** @type {NerdamerSymbolType} */ (_.parse(term)))
                );
                const expr = new core.Expression(term);
                return expr;
            });
            return result;
        } finally {
            core.Utils.disarmTimeout();
        }
    };

    core.Expression.prototype.expand = function expand() {
        if (this.symbol instanceof Equation) {
            const clone = this.symbol.clone();
            clone.RHS = /** @type {NerdamerSymbolType} */ (_.expand(clone.RHS));
            clone.LHS = /** @type {NerdamerSymbolType} */ (_.expand(clone.LHS));
            return new core.Expression(clone);
        }
        return new core.Expression(_.expand(/** @type {NerdamerSymbolType} */ (this.symbol)));
    };

    // eslint-disable-next-line func-names -- naming this 'variables' would shadow the imported variables utility
    core.Expression.prototype.variables = function () {
        if (this.symbol instanceof Equation) {
            return core.Utils.arrayUnique(
                core.Utils.variables(this.symbol.LHS).concat(core.Utils.variables(this.symbol.RHS))
            );
        }
        return core.Utils.variables(this.symbol);
    };

    const setEq = function setEq(a, b) {
        return _.equals(a, b);
    };

    // Link the Equation class back to the core
    core.Equation = Equation;

    // Loops through an array and attempts to fails a test. Stops if manages to fail.
    const checkAll = (core.Utils.checkAll = function checkAll(args, test) {
        for (let i = 0; i < args.length; i++) {
            if (test(args[i])) {
                return false;
            }
        }
        return true;
    });

    // Version solve
    /** @type {SolveModuleType} */
    const __ = (core.Solve = {
        version: '2.0.3',
        /** @type {NerdamerSymbolType[]} */
        solutions: [],
        solve(eq, variable) {
            const save = Settings.PARSE2NUMBER;
            Settings.PARSE2NUMBER = false;
            const solution = solve(eq, String(variable));
            Settings.PARSE2NUMBER = save;
            return new core.Vector(solution);
            // Return new core.Vector(solve(eq.toString(), variable ? variable.toString() : variable));
        },
        /**
         * Brings the equation to LHS. A string can be supplied which will be converted to an Equation
         *
         * @param {Equation | string | NerdamerSymbolType} eqn
         * @param {boolean} [expand]
         * @returns {NerdamerSymbolType}
         */
        toLHS(eqn, expand) {
            if (isSymbol(eqn)) {
                return eqn;
            }
            // If it's an equation then call its toLHS function instead
            if (!(eqn instanceof Equation)) {
                const eqnStr = /** @type {string} */ (eqn);
                const es = eqnStr.split('=');
                // Convert falsey values to zero
                es[1] ||= '0';
                eqn = new Equation(
                    /** @type {NerdamerSymbolType} */ (_.parse(es[0])),
                    /** @type {NerdamerSymbolType} */ (_.parse(es[1]))
                );
            }
            return eqn.toLHS(expand);
        },
        //        GetSystemVariables: function(eqns) {
        //            vars = variables(eqns[0], null, null, true);
        //
        //            //get all variables
        //            for (let i = 1, l=eqns.length; i < l; i++)
        //                vars = vars.concat(variables(eqns[i]));
        //            //remove duplicates
        //            vars = core.Utils.arrayUnique(vars).sort();
        //
        //            //done
        //            return vars;
        //        },
        /**
         * Solve a set of circle equations.
         *
         * @param {NerdamerSymbolType[]} eqns
         * @param {string[]} vars
         * @returns {Array | object}
         */
        solveCircle(eqns, vars) {
            // Convert the variables to symbols
            const svars = vars.map(x => /** @type {NerdamerSymbolType} */ (_.parse(x)));

            /** @type {number[][]} */
            const deg = [];

            /** @type {CircleSolutionResultType} */
            let solutions = [];

            // Get the degree for the equations
            for (let i = 0; i < eqns.length; i++) {
                /** @type {number[]} */
                const d = [];
                for (let j = 0; j < svars.length; j++) {
                    d.push(Number(_A.degree(eqns[i], svars[j])));
                }
                // Store the total degree
                d.push(/** @type {number} */ (core.Utils.arraySum(d, true)));
                deg.push(d);
            }

            let a = eqns[0];
            let b = eqns[1];

            if (deg[0][2] > deg[1][2]) {
                [b, a] = [a, b];
                [deg[1], deg[0]] = [deg[0], deg[1]];
            }

            // Only solve it's truly a circle
            if (deg[0][0] === 1 && deg[0][2] === 2 && deg[1][0] === 2 && deg[1][2] === 4) {
                // For clarity we'll refer to the variables as x and y
                const x = vars[0];
                const y = vars[1];

                // We can now get the two points for y
                const yPoints = solve(
                    /** @type {NerdamerSymbolType} */ (
                        _.parse(b, knownVariable(x, solve(/** @type {NerdamerSymbolType} */ (_.parse(a)), x)[0]))
                    ),
                    y
                ).map(pt => pt.toString());

                // Since we now know y we can get the two x points from the first equation
                const xPoints = [
                    solve(/** @type {NerdamerSymbolType} */ (_.parse(a, knownVariable(y, yPoints[0]))))[0].toString(),
                ];

                if (yPoints[1]) {
                    xPoints.push(
                        solve(
                            /** @type {NerdamerSymbolType} */ (_.parse(a, knownVariable(y, yPoints[1])))
                        )[0].toString()
                    );
                }

                if (Settings.SOLUTIONS_AS_OBJECT) {
                    /** @type {Record<string, string[]>} */
                    const solObj = {};
                    solObj[x] = xPoints;
                    solObj[y] = yPoints;
                    solutions = solObj;
                } else {
                    yPoints.unshift(y);
                    xPoints.unshift(x);
                    solutions = [xPoints, yPoints];
                }
            }

            return solutions;
        },
        /**
         * Solve a system of nonlinear equations
         *
         * @param {NerdamerSymbolType[]} eqns The array of equations
         * @param {number} [tries] The maximum number of tries
         * @param {number} [start] The starting point where to start looking for solutions
         * @returns {SystemSolutionResultType | []}
         */
        solveNonLinearSystem(eqns, tries, start) {
            if (tries < 0) {
                return []; // Can't find a solution
            }

            start = typeof start === 'undefined' ? core.Settings.NON_LINEAR_START : start;

            // The maximum number of times to jump
            const maxTries = core.Settings.MAX_NON_LINEAR_TRIES;

            // Halfway through the tries
            const halfway = Math.floor(maxTries / 2);

            // Initialize the number of tries to 10 if not specified
            tries = typeof tries === 'undefined' ? maxTries : tries;

            // A point at which we check to see if we're converging. By inspection it seems that we can
            // use around 20 iterations to see if we're converging. If not then we retry a jump of x
            const jumpAt = core.Settings.NON_LINEAR_JUMP_AT;

            // We jump by this many points at each pivot point
            const jump = core.Settings.NON_LINEAR_JUMP_SIZE;

            // Used to check if we actually found a solution or if we gave up. Assume we will find a solution.
            let found = true;

            const createSubs = function (vars, matrix) {
                return vars.map((x, i) => Number(matrix.get(i, 0)));
            };

            const vars = core.Utils.arrayGetVariables(eqns);
            const jacobian = core.Matrix.jacobian(eqns, vars, x => build(x, vars), true);

            const maxIter = core.Settings.MAX_NEWTON_ITERATIONS;
            let o;
            let y;
            let iters;
            let xn1;
            let norm;
            let lnorm;
            let xn;
            let d;

            const fEqns = eqns.map(eq => build(eq, vars));

            // Note: J stores compiled functions, not symbols. We use Matrix for its iteration
            // capabilities, but elements are actually compiled functions `(...args: number[]) => number`
            // The type system expects NerdamerSymbol but we're deliberately storing functions.
            const J = jacobian.map(
                (/** @type {NerdamerSymbolType} */ e) =>
                    /** @type {NerdamerSymbolType} */ (/** @type {unknown} */ (build(e, vars))),
                true
            );
            // Initial values
            xn1 = core.Matrix.cMatrix(0, vars);

            // Initialize the c matrix with something close to 0.
            let c = core.Matrix.cMatrix(start, vars);

            iters = 0;

            // Start of algorithm
            do {
                // If we've reached the max iterations then exit
                if (iters > maxIter) {
                    found = false;
                    break;
                }

                // Set the substitution object
                o = createSubs(vars, c);

                // Set xn
                xn = c.clone();

                // Capture current values for use in callbacks
                const currentO = o;
                const currentC = c;

                // Make all the substitutions for each of the equations
                fEqns.forEach((f, i) => {
                    currentC.set(i, 0, f(...currentO));
                });

                let m = new core.Matrix();
                // J actually contains compiled functions, cast to access them
                /** @type {{ each: (fn: (element: unknown, row: number, col: number) => void) => void }} */ (
                    /** @type {unknown} */ (J)
                ).each((fn, i, j) => {
                    const ans = /** @type {(...args: number[]) => number} */ (fn)(...currentO);
                    m.set(i, j, ans);
                });

                m = m.invert();

                // Preform the elimination
                y = /** @type {MatrixType} */ (_.multiply(m, c)).negate();

                // The callback is to avoid overflow in the coeffient denonimator
                // it converts it to a decimal and then back to a fraction. Some precision
                // is lost be it's better than overflow.
                d = y.subtract(xn1, x => _.parse(Number(x)));

                xn1 = xn.add(y, x => _.parse(Number(x)));

                // Move c is now xn1
                c = xn1;

                // Get the norm

                // the expectation is that we're converging to some answer as this point regardless of where we start
                // this may have to be adjusted at some point because of erroneous assumptions
                if (iters >= jumpAt) {
                    // Check the norm. If the norm is greater than one then it's time to try another point
                    if (Number(norm) > 1) {
                        // Reset the start point at halway
                        if (tries === halfway) {
                            start = 0;
                        }
                        const sign = tries > halfway ? 1 : -1; // Which side are we incrementing
                        // we increment +n at one side and -n at the other.
                        const n = (tries % Math.floor(halfway)) + 1;
                        // Adjust the start point
                        start += sign * n * jump;
                        // Call restart
                        return __.solveNonLinearSystem(eqns, --tries, start);
                    }
                }
                lnorm = norm;
                iters++;
                norm = d.max();

                // Exit early. Revisit if we get bugs
                if (Number(norm) === Number(lnorm)) {
                    break;
                }
            } while (Number(norm) >= Number.EPSILON);

            // Return a blank set if nothing was found;
            if (!found) {
                return [];
            }

            // Return c since that's the answer
            return /** @type {SystemSolutionResultType | []} */ (
                __.systemSolutions(c, vars, true, x => core.Utils.round(Number(x), 14))
            );
        },
        /**
         * Converts solution results to the appropriate format based on Settings.SOLUTIONS_AS_OBJECT.
         *
         * @param {MatrixType} result The result matrix
         * @param {string[]} vars The variable names
         * @param {boolean} [expandResult] Whether to expand the result
         * @param {Function} [callback] Optional callback to transform each solution value
         * @returns {SystemSolutionResultType}
         */
        systemSolutions(result, vars, expandResult, callback) {
            if (core.Settings.SOLUTIONS_AS_OBJECT) {
                /** @type {Record<string, SystemSolutionValueType>} */
                const solutions = {};
                result.each((e, idx) => {
                    /** @type {SystemSolutionValueType} */
                    let solution = /** @type {string | number} */ ((expandResult ? _.expand(e) : e).valueOf());
                    if (callback) {
                        solution = callback.call(e, solution);
                    }
                    solutions[vars[idx]] = solution;
                });
                return solutions;
            }
            /** @type {[string, SystemSolutionValueType][]} */
            const solutions = [];
            result.each((e, idx) => {
                /** @type {SystemSolutionValueType} */
                let solution = /** @type {string | number} */ ((expandResult ? _.expand(e) : e).valueOf());
                if (callback) {
                    solution = callback.call(e, solution);
                }
                solutions.push([vars[idx], solution]);
            });
            return solutions;
        },
        /**
         * Solves a system of equations by substitution. This is useful when no distinct solution exists. e.g. a line,
         * plane, etc.
         *
         * @param {Array} eqns
         * @returns {CircleSolutionResultType | []}
         */
        solveSystemBySubstitution(eqns) {
            // Assume at least 2 equations. The function variables will just return an empty array if undefined is provided
            const varsA = variables(eqns[0]);
            const varsB = variables(eqns[1]);
            // Check if it's a circle equation
            if (eqns.length === 2 && varsA.length === 2 && core.Utils.arrayEqual(varsA, varsB)) {
                return /** @type {CircleSolutionResultType | []} */ (__.solveCircle(eqns, varsA));
            }

            return []; // Return an empty set
        },

        // https://www.lakeheadu.ca/sites/default/files/uploads/77/docs/RemaniFinal.pdf
        /**
         * Solves a systems of equations
         *
         * @param {Array} eqns An array of equations
         * @param {Array} varArray An array of variables
         * @returns {Array | object}
         */
        solveSystem(eqns, varArray) {
            // Check if a varArray was specified
            // nerdamer.clearVars();// this deleted ALL variables: not what we want
            // parse all the equations to LHS. Remember that they come in as strings
            for (let i = 0; i < eqns.length; i++) {
                eqns[i] = __.toLHS(eqns[i]);
            }

            const l = eqns.length;
            let m = new core.Matrix();
            const c = new core.Matrix();
            let expandResult = false;
            let vars;

            if (typeof varArray === 'undefined') {
                // Check to make sure that all the equations are linear
                if (!_A.allLinear(eqns)) {
                    try {
                        return __.solveNonLinearSystem(eqns);
                    } catch (e) {
                        if (e.message === 'timeout') {
                            throw e;
                        }
                        if (e instanceof core.exceptions.DivisionByZero) {
                            return __.solveSystemBySubstitution(eqns);
                        }
                    }
                }

                vars = core.Utils.arrayGetVariables(eqns);

                // If the system only has one variable then we solve for the first one and
                // then test the remaining equations with that solution. If any of the remaining
                // equation fails then the system has no solution
                if (vars.length === 1) {
                    let n = 0;
                    let sol;
                    let e;
                    do {
                        e = eqns[n].clone();

                        if (n > 0) {
                            e = e.sub(vars[0], sol[0]);
                        }

                        sol = solve(e, vars[0]);
                        // Skip the first one
                        if (n === 0) {
                            continue;
                        }
                    } while (++n < eqns.length);

                    // Format the output
                    let solutions;
                    if (Settings.SOLUTIONS_AS_OBJECT) {
                        solutions = {};
                        solutions[vars[0]] = sol;
                    } else if (sol.length === 0) {
                        solutions = sol; // No solutions
                    } else {
                        solutions = [vars[0], sol];
                    }

                    return solutions;
                }

                // Deal with redundant equations as expressed in #562
                // The fix is to remove all but the number of equations equal to the number
                // of variables. We then solve those and then evaluate the remaining equations
                // with those solutions. If the all equal true then those are just redundant
                // equations and we can return the solution set.
                if (vars.length < eqns.length) {
                    const reduced = [];
                    const n = eqns.length;
                    for (let i = 0; i < n - 1; i++) {
                        reduced.push(_.parse(eqns[i]));
                    }

                    /** @type {Record<string, NerdamerSymbolType | string | number>} */
                    const knowns = {};
                    const solutions = __.solveSystem(reduced, vars);
                    // The solutions may have come back as an array
                    if (Array.isArray(solutions)) {
                        solutions.forEach(sol => {
                            // For substitution, we only use single-value solutions (not arrays)
                            if (!Array.isArray(sol[1])) {
                                knowns[sol[0]] = sol[1];
                            }
                        });
                    } else {
                        // Filter out array solutions for substitution
                        for (const key of Object.keys(solutions)) {
                            const val = solutions[key];
                            if (!Array.isArray(val)) {
                                knowns[key] = val;
                            }
                        }
                    }

                    // Start by assuming they will all evaluate to zero. If even one fails
                    // then all zero will be false
                    let allZero = true;
                    // Check if the last solution evalutes to zero given these solutions
                    for (let i = n - 1; i < n; i++) {
                        if (!(/** @type {NerdamerSymbolType} */ (_.parse(eqns[i], knowns)).equals(0))) {
                            allZero = false;
                        }
                    }

                    if (allZero) {
                        return solutions;
                    }
                }

                // Deletes only the variables of the linear equations in the nerdamer namespace
                for (let i = 0; i < vars.length; i++) {
                    nerdamer.setVar(vars[i], 'delete');
                }
                // TODO: move this to cMatrix or something similar
                // populate the matrix
                for (let i = 0; i < l; i++) {
                    const e = eqns[i]; // Store the expression
                    // Iterate over the columns
                    for (let j = 0; j < vars.length; j++) {
                        const v = vars[j];
                        let coeffs = [];
                        e.each(x => {
                            if (x.contains(v)) {
                                coeffs = coeffs.concat(x.coeffs());
                            }
                        });

                        const cf = core.Utils.arraySum(coeffs);
                        m.set(i, j, cf);
                    }

                    // Strip the variables from the symbol so we're left with only the zeroth coefficient
                    // start with the symbol and remove each variable and its coefficient
                    let num = e.clone();
                    vars.forEach(varName => {
                        num = num.stripVar(varName, true);
                    });
                    c.set(i, 0, num.negate());
                }
            } else {
                /**
                 * The idea is that we loop through each equation and then expand it. Afterwards we loop through each
                 * term and see if and check to see if it matches one of the variables. When a match is found we mark
                 * it. No other match should be found for that term. If it is we stop since it's not linear.
                 */
                vars = varArray;
                expandResult = true;
                for (let i = 0; i < l; i++) {
                    // Prefill
                    c.set(i, 0, new NerdamerSymbol(0));
                    const e = /** @type {NerdamerSymbolType[]} */ (
                        /** @type {NerdamerSymbolType} */ (_.expand(eqns[i])).collectSummandSymbols()
                    ); // Expand and store
                    // go trough each of the variables
                    for (let j = 0; j < varArray.length; j++) {
                        m.set(i, j, new NerdamerSymbol(0));
                        const v = varArray[j];
                        // Go through the terms and sort the variables
                        for (let k = 0; k < e.length; k++) {
                            const term = e[k];
                            let check = false;
                            for (let z = 0; z < varArray.length; z++) {
                                // Check to see if terms contain multiple variables
                                if (term.contains(varArray[z])) {
                                    if (check) {
                                        core.Utils.err(`Multiple variables found for term ${term}`);
                                    }
                                    check = true;
                                }
                            }
                            // We made sure that every term contains one variable so it's safe to assume that if the
                            // variable is found then the remainder is the coefficient.
                            if (term.contains(v)) {
                                const tparts = /** @type {(NerdamerSymbolType | VectorType | MatrixType)[]} */ (
                                    explode(remove(e, k), v)
                                );
                                k--; // Issue #52: decrement k to hit this spot in the array e again next loop
                                m.set(i, j, _.add(m.get(i, j), /** @type {NerdamerSymbolType} */ (tparts[0])));
                            }
                        }
                    }
                    // All the remaining terms go to the c matrix
                    for (let k = 0; k < e.length; k++) {
                        c.set(i, 0, _.add(c.get(i, 0), e[k]));
                    }
                }
                // Consider case (a+b)*I+u
            }

            // Check if the system has a distinct solution
            if (vars.length !== eqns.length || m.determinant().equals(0)) {
                // Solve the system by hand
                // return __.solveSystemBySubstitution(eqns, vars, m, c);
                throw new core.exceptions.SolveError('System does not have a distinct solution');
            }

            // Use M^-1*c to solve system
            m = m.invert();
            const result = m.multiply(c);
            // Correct the sign as per issue #410
            if (core.Utils.isArray(varArray)) {
                result.each(x => x.negate());
            }

            return __.systemSolutions(result, vars, expandResult);
        },
        /**
         * The quadratic function but only one side.
         *
         * @param {NerdamerSymbolType} c
         * @param {NerdamerSymbolType} b
         * @param {NerdamerSymbolType} a
         * @returns {(NerdamerSymbolType | VectorType | MatrixType)[]}
         */
        quad(c, b, a) {
            let discriminant = _.subtract(
                _.pow(b.clone(), new NerdamerSymbol(2)),
                _.multiply(_.multiply(a.clone(), c.clone()), new NerdamerSymbol(4))
            ); /* B^2 - 4ac*/
            // Fix for #608
            discriminant = /** @type {NerdamerSymbolType} */ (_.expand(discriminant));
            const det = /** @type {NerdamerSymbolType} */ (_.pow(discriminant, new NerdamerSymbol(0.5)));
            const den = /** @type {NerdamerSymbolType} */ (
                _.parse(/** @type {NerdamerSymbolType} */ (_.multiply(new NerdamerSymbol(2), a.clone())))
            );
            const retval = [
                _.parse(format('(-({0})+({1}))/({2})', b, det, den)),
                _.parse(format('(-({0})-({1}))/({2})', b, det, den)),
            ];

            return retval;
        },
        /**
         * The cubic equation
         * http://math.stackexchange.com/questions/61725/is-there-a-systematic-way-of-solving-cubic-equations
         *
         * @param {NerdamerSymbolType} dO
         * @param {NerdamerSymbolType} cO
         * @param {NerdamerSymbolType} bO
         * @param {NerdamerSymbolType} aO
         * @returns {Array}
         */
        cubic(dO, cO, bO, aO) {
            // Convert everything to text
            const a = aO.text();
            const b = bO.text();
            const c = cO.text();
            const d = dO.text();

            const t = `(-(${b})^3/(27*(${a})^3)+(${b})*(${c})/(6*(${a})^2)-(${d})/(2*(${a})))`;
            const u = `((${c})/(3*(${a}))-(${b})^2/(9*(${a})^2))`;
            const v = `(${b})/(3*(${a}))`;
            const x = `((${t})+sqrt((${t})^2+(${u})^3))^(1/3)+((${t})-sqrt((${t})^2+(${u})^3))^(1/3)-(${v})`;

            // Convert a to one
            const w = '1/2+sqrt(3)/2*i'; // Cube root of unity

            return [_.parse(x), _.parse(`(${x})(${w})`), _.parse(`(${x})(${w})^2`)];

            // https://www.wikihow.com/Solve-a-Cubic-Equation method 3
            // const delta0 = `(${b})^2-(3*(${a})(${c}))`;
            // _.parse(delta0);
            // const delta1 = `2(${b})^3-(9*(${a})(${b})(${c}))+27((${a})^2)(${d})`;
            // _.parse(delta1);
            // // const delta = `(${delta1})^2-(4*(${delta0})^3)/(-27(${a})^2)`;

            // const C = `((sqrt((${delta1})^2-(4*(${delta0})^3))+(${delta1}))/2)^(1/3)`;
            // _.parse(C);
            // const u = `(-1+sqrt(-3))/2`
            // _.parse(u);

            // const result = []
            // for (let n = 1; n <=3; n++) {
            //     let x = `-((${b})+ (${u})^${n}*(${C})+(${delta0})/((${u})^${n}*(${C})))/(3(${a}))`;
            //     console.log(x);
            //     console.log(x.substring(168));
            //     result.push(_.parse(x));
            // }

            // return result.map((x)=>_.parse(x))
        },
        /**
         * The quartic equation
         *
         * @param {NerdamerSymbolType} e
         * @param {NerdamerSymbolType} d
         * @param {NerdamerSymbolType} c
         * @param {NerdamerSymbolType} b
         * @param {NerdamerSymbolType} a
         * @returns {Array}
         */
        quartic(e, d, c, b, a) {
            /** @type {Record<string, number>} */
            const scope = {};
            core.Utils.arrayUnique(
                variables(a).concat(variables(b)).concat(variables(c)).concat(variables(d)).concat(variables(e))
            ).forEach(x => {
                scope[x] = 1;
            });
            const aStr = a.toString();
            const bStr = b.toString();
            const cStr = c.toString();
            const dStr = d.toString();
            const eStr = e.toString();
            let _D;
            /* Var D = core.Utils.block('PARSE2NUMBER', function() {
             return _.parse(format("256*({0})^3*({4})^3-192*({0})^2*({1})*({3})*({4})^2-128*({0})^2*({2})^2*({4})^2+144*({0})^2*({2})*({3})^2*({4})"+
             "-27*({0})^2*({3})^4+144*({0})*({1})^2*({2})*({4})^2-6*({0})*({1})^2*({3})^2*({4})-80*({0})*({1})*({2})^2*({3})*({4})+18*({0})*({1})*({2})*({3})^3"+
             "+16*({0})*({2})^4*({4})-4*({0})*({2})^3*({3})^2-27*({1})^4*({4})^2+18*({1})^3*({2})*({3})*({4})-4*({1})^3*({3})^3-4*({1})^2*({2})^3*({4})+({1})^2*({2})^2*({3})^2", 
             aStr, bStr, cStr, dStr, eStr), scope);
             });*/

            const p = _.parse(format('(8*({0})*({2})-3*({1})^2)/(8*({0})^2)', aStr, bStr, cStr)).toString(); // A, b, c
            const q = _.parse(
                format('(({1})^3-4*({0})*({1})*({2})+8*({0})^2*({3}))/(8*({0})^3)', aStr, bStr, cStr, dStr)
            ).toString(); // A, b, c, d, e
            const D0 = _.parse(format('12*({0})*({4})-3*({1})*({3})+({2})^2', aStr, bStr, cStr, dStr, eStr)).toString(); // A, b, c, d, e
            const D1 = _.parse(
                format(
                    '2*({2})^3-9*({1})*({2})*({3})+27*({1})^2*({4})+27*({0})*({3})^2-72*({0})*({2})*({4})',
                    aStr,
                    bStr,
                    cStr,
                    dStr,
                    eStr
                )
            ).toString(); // A, b, c, d, e
            const Q = _.parse(format('((({1})+(({1})^2-4*({0})^3)^(1/2))/2)^(1/3)', D0, D1)).toString(); // D0, D1
            const quarticS = _.parse(
                format('(1/2)*(-(2/3)*({1})+(1/(3*({0}))*(({2})+(({3})/({2})))))^(1/2)', aStr, p, Q, D0)
            ).toString(); // A, p, Q, D0
            const x1 = _.parse(
                format(
                    '-(({1})/(4*({0})))-({4})+(1/2)*sqrt(-4*({4})^2-2*({2})+(({3})/({4})))',
                    aStr,
                    bStr,
                    p,
                    q,
                    quarticS
                )
            ); // A, b, p, q, S
            const x2 = _.parse(
                format(
                    '-(({1})/(4*({0})))-({4})-(1/2)*sqrt(-4*({4})^2-2*({2})+(({3})/({4})))',
                    aStr,
                    bStr,
                    p,
                    q,
                    quarticS
                )
            ); // A, b, p, q, S
            const x3 = _.parse(
                format(
                    '-(({1})/(4*({0})))+({4})+(1/2)*sqrt(-4*({4})^2-2*({2})-(({3})/({4})))',
                    aStr,
                    bStr,
                    p,
                    q,
                    quarticS
                )
            ); // A, b, p, q, S
            const x4 = _.parse(
                format(
                    '-(({1})/(4*({0})))+({4})-(1/2)*sqrt(-4*({4})^2-2*({2})-(({3})/({4})))',
                    aStr,
                    bStr,
                    p,
                    q,
                    quarticS
                )
            ); // A, b, p, q, S
            return [x1, x2, x3, x4];
        },
        /**
         * Breaks the equation up in its factors and tries to solve the smaller parts
         *
         * @param {NerdamerSymbolType} symbol
         * @param {string} solveFor
         * @returns {Array}
         */
        divideAndConquer(symbol, solveFor) {
            let sols = [];
            // See if we can solve the factors
            const factors = Factor.factorInner(symbol);
            if (factors.group === CB) {
                factors.each(x => {
                    x = NerdamerSymbol.unwrapPARENS(x);
                    sols = sols.concat(solve(x, solveFor));
                });
            }
            return sols;
        },
        /**
         * Attempts to solve the equation assuming it's a polynomial with numeric coefficients
         *
         * @param {NerdamerSymbolType} eq
         * @param {string} solveFor
         * @returns {Array}
         */
        csolve(eq, solveFor) {
            return core.Utils.block(
                'IGNORE_E',
                () => {
                    let p;
                    let pn;
                    let n;
                    let pf;
                    let r;
                    let _theta;
                    let sr;
                    let _sp;
                    const roots = [];
                    const f = /** @type {DecomposeResultType} */ (core.Utils.decompose_fn(eq, solveFor, true));
                    if (f.x.group === S) {
                        p = _.parse(f.x.power);
                        pn = Number(p);
                        n = _.pow(_.divide(f.b.negate(), f.a), /** @type {NerdamerSymbolType} */ (p).invert());
                        pf = NerdamerSymbol.toPolarFormArray(/** @type {NerdamerSymbolType} */ (n));
                        r = pf[0];
                        _theta = pf[1];
                        sr = r.toString();
                        _sp = p.toString();
                        let k;
                        let root;
                        let str;
                        for (let i = 0; i < pn; i++) {
                            k = i;
                            str = format('({0})*e^(2*{1}*pi*{2}*{3})', sr, k, p, core.Settings.IMAGINARY);
                            root = _.parse(str);
                            roots.push(root);
                        }
                    }
                    return roots;
                },
                true
            );
        },
        /**
         * Generates starting points for the Newton solver given an expression at zero. It begins by checking if zero is
         * a good point and starts expanding by a provided step size. Builds on the fact that if the sign changes over
         * an interval then a zero must exist on that interval
         *
         * @param {NerdamerSymbolType} symbol
         * @param {number} step
         * @param {boolean} extended
         * @returns {Array}
         */
        getPoints(symbol, step, extended) {
            step ||= 0.01;
            let points = [];
            const f = build(symbol);
            const x0 = 0;

            const start = Math.round(x0);
            const _last = f(start);
            const rside = core.Settings.ROOTS_PER_SIDE; // The max number of roots on right side
            const lside = rside; // The max number of roots on left side
            // check around the starting point
            points.push(Math.floor(start / 2)); // Half way from zero might be a good start
            points.push(Math.abs(start)); // |f(0)| could be a good start
            points.push(start); // |f(0)| could be a good start
            // adjust for log. A good starting point to include for log is 0.1
            symbol.each(x => {
                if (x.containsFunction(core.Settings.LOG)) {
                    points.push(0.1);
                }
            });

            const left = range(-core.Settings.SOLVE_RADIUS, start, step);
            const right = range(start, core.Settings.SOLVE_RADIUS, step);

            const testSide = function (side, numRoots) {
                // Console.log("test side "+side[0]+":"+side.at(-1));
                let xi;
                let val;
                let sign;
                const hits = [];
                const lastPoint = side[0];
                let lastSign = Math.sign(f(lastPoint));
                for (let i = 0, l = side.length; i < l && hits.length < numRoots; i++) {
                    xi = side[i]; // The point being evaluated
                    val = f(xi);
                    sign = Math.sign(val);
                    // Don't add non-numeric values
                    if (isNaN(sign)) {
                        continue;
                    }

                    // Compare the signs. The have to be different if they cross a zero
                    if (sign !== lastSign) {
                        hits.push(xi); // Take note of the possible zero location
                        hits.push(side[i - 1]); // Also the other side
                        // console.log("   hit at "+xi);
                        // if (hits.length >= numRoots){
                        //     break;
                        // }
                    }
                    lastSign = sign;
                }

                points = points.concat(hits);
            };

            testSide(left, lside);
            testSide(right, rside);

            if (extended) {
                // Check for sign changes way outside the range
                // in a limited way
                const max = core.Settings.SOLVE_RADIUS;
                testSide([max, max * max], 1);
                testSide([-max * max, -max], 1);
            }

            // Console.log("points: "+points);
            return points;
        },
        /**
         * Implements the bisection method. Returns undefined in no solution is found
         *
         * @param {number} point
         * @param {Function} f
         * @returns {undefined | number}
         */
        bisection(point, f) {
            let left = point - 1;
            let right = point + 1;
            // First test if this point is even worth evaluating. It should
            // be crossing the x axis so the signs should be different
            if (Math.sign(f(left)) !== Math.sign(f(right))) {
                let safety = 0;

                let epsilon;
                let middle;

                do {
                    epsilon = Math.abs(right - left);
                    // Safety against an infinite loop
                    if (safety++ > core.Settings.MAX_BISECTION_ITER || isNaN(epsilon)) {
                        return undefined;
                    }
                    // Calculate the middle point
                    middle = (left + right) / 2;

                    if (f(left) * f(middle) > 0) {
                        left = middle;
                    } else {
                        right = middle;
                    }
                } while (epsilon >= Settings.EPSILON);

                const solution = (left + right) / 2;

                // Test the solution to make sure that it's within tolerance
                const xPoint = f(solution);

                if (!isNaN(xPoint) && Math.abs(xPoint) <= core.Settings.BI_SECTION_EPSILON) {
                    // Returns too many junk solutions if not rounded at 13th place.
                    return /** @type {number} */ (core.Utils.round(solution, 13));
                }
                return undefined;
            }
            return undefined;
        },
        // Helper function for when Newton gets into the weeds
        // look from left and right of a sign-change interval
        // narrows it down
        // result: A real point with tractable numbers to continue from
        // or undefined
        bSearch(left, right, f) {
            let fLeft = f(left);
            let fRight = f(right);
            // Reject imposters
            if (Math.sign(fLeft) === Math.sign(fRight) || isNaN(fLeft) || isNaN(fRight)) {
                return undefined;
            }

            const maxIter = 80; // Guess the amount of iterations to outrun precision?
            let iterations = 0;
            do {
                const x = (left + right) / 2;
                const sLeft = Math.sign(fLeft);
                const sX = Math.sign(f(x));
                if (sLeft === sX) {
                    if (x === left) {
                        break; // Precision exceeded
                    }
                    left = x;
                    fLeft = f(left);
                } else {
                    if (x === right) {
                        break; // Precision exceeded
                    }
                    right = x;
                    fRight = f(right);
                }
                iterations++;
            } while (left !== right && iterations < maxIter);
            // If one of them is infinite or NaN, there is probably a singularity here
            if (!isFinite(f(left)) || !isFinite(f(right))) {
                return undefined;
            }
            // Return the point where the absolute value is smaller
            // return (Math.abs(f(left)) < Math.abs(f(right)))? left:right;
            return left;
        },
        /**
         * Implements Newton's iterations. Returns undefined if no solutions if found
         *
         * @param {number} point
         * @param {Function} f
         * @param {Function} fp
         * @returns {undefined | number}
         */
        Newton(point, f, fp, point2) {
            // Console.log("Newton point "+point);
            const maxiter = core.Settings.MAX_NEWTON_ITERATIONS;
            let iter = 0;
            // First try the point itself. If it's zero voila. We're done
            let x0 = point;
            let x;
            let e;
            let delta;
            do {
                const fx0 = f(x0); // Store the result of the function
                // if the value is zero then we're done because 0 - (0/d f(x0)) = 0
                if (x0 === 0 && fx0 === 0) {
                    x = 0;
                    // Console.log("  exact zero");
                    break;
                }

                iter++;
                if (iter > maxiter) {
                    // Console.log("   iter:"+iter+", last e:"+e);
                    return undefined;
                }

                const fpx0 = fp(x0);
                // Infinite or NaN or 0 derivative at x0?
                if (isNaN(fpx0) || isNaN(fx0)) {
                    // Nothing we can do
                    // console.log("   non-finite derivative");
                    return undefined;
                }
                if (fpx0 === 0) {
                    // Max/min or saddle point. what can we do? repeat last delta.
                    x += delta;
                } else if (!isFinite(fx0) || !isFinite(fpx0) || Math.abs(fx0) > 1e25) {
                    // Hail Mary: binary search through the
                    // sign-switch interval
                    return __.bSearch(point2, x0, /** @type {(x: number) => number} */ (f));
                    // // numbers got too big
                    // // at least follow the slope down
                    // const direction = Math.sign(fpx0)/Math.sign(fx0);
                    // // direction is 1 or -1
                    // // big and growing: shrink x
                    // // -big and -growing: shrink x
                    // // big and -growing: grow x
                    // // -big and growing: grow x
                    // if (x0 === 0) {
                    //     // just move it a bit, so in the next loop
                    //     // we can make progress
                    //     x = x0 + direction;
                    // } else {
                    //     // can shrink/grow by dividing or multiplying
                    //     x = x0 / (direction===1?2:0.5);
                    // }
                } else {
                    // Regular case, follow tangent
                    x = x0 - fx0 / fpx0;
                    // Console.log("new x: "+x);
                }
                delta = x - x0;
                if (delta === 0 && !isFinite(fpx0)) {
                    // No movement
                    return undefined;
                }
                e = Math.abs(delta);
                x0 = x;
            } while (e > Settings.NEWTON_EPSILON);

            // Console.log("   found "+x);
            return x;
        },
        rewrite(rhs, lhs, forVariable) {
            lhs ||= new NerdamerSymbol(0);
            if (rhs.isComposite() && rhs.isLinear()) {
                // Try to isolate the square root
                // container for the square roots
                const sqrts = [];
                // All else
                const rem = [];
                rhs.each(x => {
                    x = x.clone();
                    if (x.fname === 'sqrt' && x.contains(forVariable)) {
                        sqrts.push(x);
                    } else {
                        rem.push(x);
                    }
                }, true);

                if (sqrts.length === 1) {
                    // Move the remainder to the RHS
                    lhs = /** @type {NerdamerSymbolType} */ (
                        _.expand(
                            _.pow(
                                _.subtract(lhs, /** @type {NerdamerSymbolType} */ (core.Utils.arraySum(rem))),
                                new NerdamerSymbol(2)
                            )
                        )
                    );
                    // Square both sides
                    rhs = /** @type {NerdamerSymbolType} */ (
                        _.expand(_.pow(NerdamerSymbol.unwrapSQRT(sqrts[0]), new NerdamerSymbol(2)))
                    );
                }
            } else {
                rhs = NerdamerSymbol.unwrapSQRT(/** @type {NerdamerSymbolType} */ (_.expand(rhs))); // Expand the term expression go get rid of quotients when possible
            }

            let c = 0; // A counter to see if we have all terms with the variable
            const l = rhs.length;
            // Try to rewrite the whole thing
            if (rhs.group === CP && rhs.contains(forVariable) && rhs.isLinear()) {
                rhs.distributeMultiplier();
                let t = new NerdamerSymbol(0);
                // First bring all the terms containing the variable to the lhs
                rhs.each(x => {
                    if (x.contains(forVariable)) {
                        c++;
                        t = /** @type {NerdamerSymbolType} */ (_.add(t, x.clone()));
                    } else {
                        lhs = /** @type {NerdamerSymbolType} */ (_.subtract(lhs, x.clone()));
                    }
                });
                rhs = t;

                // If not all the terms contain the variable so it's in the form
                // a*x^2+x
                if (c !== l) {
                    return __.rewrite(rhs, lhs, forVariable);
                }
                return [rhs, lhs];
            }
            if (rhs.group === CB && rhs.contains(forVariable) && rhs.isLinear()) {
                if (rhs.multiplier.lessThan(0)) {
                    rhs.multiplier = rhs.multiplier.multiply(new core.Frac(-1));
                    lhs.multiplier = lhs.multiplier.multiply(new core.Frac(-1));
                }
                if (lhs.equals(0)) {
                    return new NerdamerSymbol(0);
                }
                let t = new NerdamerSymbol(1);
                rhs.each(x => {
                    if (x.contains(forVariable)) {
                        t = /** @type {NerdamerSymbolType} */ (_.multiply(t, x.clone()));
                    } else {
                        lhs = /** @type {NerdamerSymbolType} */ (_.divide(lhs, x.clone()));
                    }
                });
                rhs = t;
                return __.rewrite(rhs, lhs, forVariable);
            }
            if (!rhs.isLinear() && rhs.contains(forVariable)) {
                const p = /** @type {NerdamerSymbolType} */ (_.parse(rhs.power.clone().invert()));
                rhs = /** @type {NerdamerSymbolType} */ (_.pow(rhs, p.clone()));
                lhs = /** @type {NerdamerSymbolType} */ (
                    _.pow(/** @type {NerdamerSymbolType} */ (_.expand(lhs)), p.clone())
                );
                return __.rewrite(rhs, lhs, forVariable);
            }
            if (rhs.group === FN || rhs.group === S || rhs.group === PL) {
                return [rhs, lhs];
            }
            return [rhs, lhs];
        },
        sqrtSolve(symbol, v) {
            let sqrts = new NerdamerSymbol(0);
            let rem = new NerdamerSymbol(0);
            if (symbol.isComposite()) {
                symbol.each(x => {
                    if (x.fname === 'sqrt' && x.contains(v)) {
                        sqrts = /** @type {NerdamerSymbolType} */ (_.add(sqrts, x.clone()));
                    } else {
                        rem = /** @type {NerdamerSymbolType} */ (_.add(rem, x.clone()));
                    }
                });
                // Quick and dirty ATM
                if (!sqrts.equals(0)) {
                    const t = _.expand(
                        _.multiply(
                            _.parse(symbol.multiplier),
                            _.subtract(_.pow(rem, new NerdamerSymbol(2)), _.pow(sqrts, new NerdamerSymbol(2)))
                        )
                    );
                    // Square both sides
                    let solutions = solve(t, v);
                    // Test the points. The dumb way of getting the answers
                    solutions = solutions.filter(e => {
                        if (e.isImaginary()) {
                            return true;
                        }
                        /** @type {Record<string, NerdamerSymbolType>} */
                        const subs = {};
                        subs[v] = e;
                        const point = evaluate(symbol, subs);
                        if (point.equals(0)) {
                            return true;
                        }
                        return false;
                    });
                    return solutions;
                }
            }
            return undefined;
        },
    });

    // Special case to handle solving equations with exactly one abs() correctly
    const absSolve = function (eqns, solveFor, depth, fn) {
        const eq = eqns.toString();
        const match = eq.match(/(?<![a-z])abs/gu);
        // Not found or more than 1 occurrence? get out!
        if (!match || match.length > 2) {
            return null;
        }
        // Can handle only abs at beginning
        if ((eqns.LHS.group !== FN || false) && eqns.RHS.group !== FN) {
            return null;
        }
        // We have exactly one abs. kill it and make two cases
        const eqplus = eqns.constructor(eq.replace(/(?<![a-z])abs/u, ''));
        const eqminus = eqns.constructor(eq.replace(/(?<![a-z])abs/u, '(-1)'));

        const resultplus = solve(eqplus, solveFor, null, depth, fn);
        const resultminus = solve(eqminus, solveFor, null, depth, fn);

        return [resultminus, resultplus];
    };
    /*
     *
     * @param {string[]|string|Equation} eqns
     * @param {string} solveFor
     * @param {Array} solutions
     * @param {number} depth
     * @param {string|Equation} fn
     * @returns {Array}
     */
    // let solve = function (eqns, solveFor, solutions, depth, fn) {
    //     let original = "<multiple>";
    //     original = eqns.toString();
    //     try {
    //         solutions = _solve(eqns, solveFor, solutions, depth, fn);
    //     } catch (error) {
    //         console.error(error);
    //     }
    //     console.log("solve: "+original+" for "+solveFor+" = "+solutions);
    //     return solutions;
    // }
    function solve(eqns, solveFor, solutions, depth, fn) {
        depth ||= 0;

        if (depth++ > Settings.MAX_SOLVE_DEPTH) {
            return solutions;
        }

        // Parse out functions. Fix for issue #300
        // eqns = core.Utils.evaluate(eqns);
        solutions ||= [];
        // Mark existing solutions as not to have duplicates
        const existing = {};

        // Easy fail. If it's a rational function and the denominator is zero
        // then we're done. Issue #555
        /** @type {Record<string, number>} */
        const known = {};
        known[solveFor] = 0;

        // Is used to add solutions to set.
        // TODO: Set is now implemented and should be utilized
        const addToResult = function (r, hasTrig) {
            const rIsSymbol = isSymbol(r);
            if (r === undefined || (typeof r === 'number' && isNaN(r))) {
                return;
            }
            if (isArray(r)) {
                r.forEach(sol => {
                    addToResult(sol);
                });
            } else if (r.valueOf() !== 'null') {
                // Call the pre-add function if defined. This could be useful for rounding
                if (typeof core.Settings.PRE_ADD_SOLUTION === 'function') {
                    r = core.Settings.PRE_ADD_SOLUTION(r);
                }

                if (!rIsSymbol) {
                    r = _.parse(r);
                }
                // Try to convert the number to multiples of pi
                if (core.Settings.make_pi_conversions && hasTrig) {
                    const temp = _.divide(r.clone(), new NerdamerSymbol(Math.PI));
                    const m = temp.multiplier;
                    const a = Math.abs(Number(m.num));
                    const b = Math.abs(Number(m.den));
                    if (a < 10 && b < 10) {
                        r = _.multiply(temp, new NerdamerSymbol('pi'));
                    }
                }

                // And check if we get a number otherwise we might be throwing out symbolic solutions.
                const rStr = r.toString();

                if (!existing[rStr]) {
                    solutions.push(r);
                }
                // Mark the answer as seen
                existing[rStr] = true;
            }
        };

        // Make preparations if it's an Equation
        if (eqns instanceof Equation) {
            // See absSolve above
            // the rest of solve does a crappy job at solving abs,
            // so we wrap it here if necessary
            const absResult = absSolve(eqns, solveFor, depth, fn);
            if (absResult) {
                addToResult(absResult);
                return solutions;
            }

            // If it's zero then we're done
            if (eqns.isZero()) {
                return [new NerdamerSymbol(0)];
            }
            // If the lhs = x then we're done
            if (eqns.LHS.equals(solveFor) && !eqns.RHS.contains(solveFor, true)) {
                return [eqns.RHS];
            }
            // If the rhs = x then we're done
            if (eqns.RHS.equals(solveFor) && !eqns.LHS.contains(solveFor, true)) {
                return [eqns.LHS];
            }
        }

        // Unwrap the vector since what we want are the elements
        if (eqns instanceof core.Vector) {
            eqns = /** @type {SolveEquationArray} */ (eqns.elements);
        }
        // If it's an array then solve it as a system of equations
        // Must check BEFORE the default assignment to preserve original solveFor value
        if (isArray(eqns)) {
            return __.solveSystem(/** @type {SolveEquationArray} */ (eqns), solveFor);
        }
        solveFor ||= 'x'; // Assumes x by default

        if (isSymbol(eqns) && evaluate(eqns.getDenom(), known).equals(0) === true) {
            return solutions;
        }

        // Maybe we get lucky. Try the point at the function. If it works we have a point
        // If not it failed
        if (eqns.group === S && eqns.contains(solveFor)) {
            try {
                /** @type {Record<string, number>} */
                const o = {};
                o[solveFor] = 0;
                evaluate(fn, o);
                addToResult(new NerdamerSymbol(0));
            } catch (e) {
                if (e.message === 'timeout') {
                    throw e;
                }
                // Do nothing;
            }

            return solutions;
        }
        if (eqns.group === CB) {
            // It suffices to solve for the numerator
            const num = eqns.getNum();

            if (num.group === CB) {
                const sf = String(solveFor); // Everything else belongs to the coeff
                // get the denominator and make sure it doesn't have x since we don't know how to solve for those
                num.each(x => {
                    if (x.contains(sf)) {
                        solve(x, solveFor, solutions, depth, eqns);
                    }
                });

                return solutions;
            }

            return solve(num, solveFor, solutions, depth, fn);
        }

        if (eqns.group === FN && eqns.fname === 'sqrt') {
            eqns = _.pow(NerdamerSymbol.unwrapSQRT(eqns), new NerdamerSymbol(2));
        }
        // Pass in false to not expand equations such as (x+y)^5.
        // It suffices to solve for the numerator since there's no value in the denominator which yields a zero for the function
        let eq = (core.Utils.isSymbol(eqns) ? eqns : __.toLHS(eqns, false)).getNum();
        const vars = core.Utils.variables(eq); // Get a list of all the variables
        const numvars = vars.length; // How many variables are we dealing with

        // it sufficient to solve (x+y) if eq is (x+y)^n since 0^n
        if (core.Utils.isInt(eq.power) && Number(eq.power) > 1) {
            eq = _.parse(eq).toLinear();
        }

        // If we're dealing with a single variable then we first check if it's a
        // polynomial (including rationals).If it is then we use the Jenkins-Traubb algorithm.
        // Don't waste time
        if ((eq.group === S || eq.group === CB) && eq.contains(solveFor)) {
            return [new NerdamerSymbol(0)];
        }
        // Force to polynomial. We go through each and then we look at what it would
        // take for its power to be an integer
        // if the power is a fractional we divide by the fractional power
        let fractionals = {};
        let cfact;

        const correctDenom = function (symbol) {
            symbol = _.expand(symbol, {
                expand_denominator: true,
                expand_functions: true,
            });
            const original = symbol.clone(); // Preserve the original

            if (symbol.symbols) {
                for (const x in symbol.symbols) {
                    if (!Object.hasOwn(symbol.symbols, x)) {
                        continue;
                    }
                    const sym = symbol.symbols[x];

                    // Get the denominator of the sub-symbol
                    const den = sym.getDenom();

                    if (!den.isConstant(true) && symbol.isComposite()) {
                        /** @type {NerdamerSymbolType} */
                        let t = new NerdamerSymbol(0);
                        symbol.each(e => {
                            t = /** @type {NerdamerSymbolType} */ (_.add(t, _.multiply(e, den.clone())));
                        });

                        return correctDenom(_.multiply(_.parse(symbol.multiplier), t));
                    }

                    const parts = explode(sym, solveFor);
                    const isSqrt = /** @type {NerdamerSymbolType} */ (parts[1]).fname === core.Settings.SQRT;
                    const v = /** @type {NerdamerSymbolType} */ (
                        NerdamerSymbol.unwrapSQRT(/** @type {NerdamerSymbolType} */ (parts[1]))
                    );
                    /** @type {FracType} */
                    const p = /** @type {FracType} */ (v.power.clone());
                    // Circular logic with sqrt. Since sqrt(x) becomes x^(1/2) which then becomes sqrt(x), this continues forever
                    // this needs to be terminated if p = 1/2
                    if (!isSymbol(p) && !p.equals(1 / 2)) {
                        if (Number(p.den) > 1) {
                            if (isSqrt) {
                                symbol = _.subtract(symbol, sym.clone());
                                symbol = _.add(symbol, _.multiply(parts[0].clone(), v));
                                return correctDenom(symbol);
                            }
                            let c = fractionals[Number(p.den)];
                            fractionals[Number(p.den)] = c ? c++ : 1;
                        } else if (p.sign() === -1) {
                            const factor = _.parse(`${solveFor}^${Math.abs(Number(p))}`); // This
                            // unwrap the symbol's denoniator
                            const currentSymbol = symbol;
                            currentSymbol.each((y, index) => {
                                if (y.contains(solveFor)) {
                                    currentSymbol.symbols[index] = _.multiply(y, factor.clone());
                                }
                            });
                            fractionals = {};
                            return correctDenom(_.parse(currentSymbol));
                        } else if (sym.group === PL) {
                            const minP = core.Utils.arrayMin(core.Utils.keys(sym.symbols).map(Number));
                            if (minP < 0) {
                                const factor = /** @type {NerdamerSymbolType} */ (
                                    _.parse(`${solveFor}^${Math.abs(minP)}`)
                                );
                                /** @type {NerdamerSymbolType} */
                                let corrected = new NerdamerSymbol(0);
                                original.each(origSym => {
                                    corrected = /** @type {NerdamerSymbolType} */ (
                                        _.add(corrected, _.multiply(origSym.clone(), factor.clone()))
                                    );
                                }, true);
                                return corrected;
                            }
                        }
                    }
                }
            }

            return symbol;
        };

        // Separate the equation
        const separate = function (equation) {
            /** @type {NerdamerSymbolType} */
            let lhs = new NerdamerSymbol(0);
            /** @type {NerdamerSymbolType} */
            let rhs = new NerdamerSymbol(0);
            equation.each(x => {
                if (x.contains(solveFor, true)) {
                    lhs = /** @type {NerdamerSymbolType} */ (_.add(lhs, x.clone()));
                } else {
                    rhs = /** @type {NerdamerSymbolType} */ (_.subtract(rhs, x.clone()));
                }
            });
            return [lhs, rhs];
        };

        /**
         * @type {(
         *     name: string,
         *     lhs: NerdamerSymbolType,
         *     rhs: NerdamerSymbolType
         * ) => NerdamerSymbolType | undefined}
         */
        __.inverseFunctionSolve = function inverseFunctionSolve(name, lhs, rhs) {
            // Ax+b comes back as [a, x, ax, b];
            const parts = explode(lhs.args[0], solveFor);
            // Check if x is by itself
            const x = /** @type {NerdamerSymbolType} */ (parts[1]);
            if (x.group === S) {
                return /** @type {NerdamerSymbolType} */ (
                    _.divide(_.symfunction(name, [_.divide(rhs, _.parse(lhs.multiplier))]), parts[0])
                );
            }
            return undefined;
        };

        // First remove any denominators
        eq = correctDenom(eq);

        if (eq.equals(0)) {
            return [eq];
        }
        // Correct fractionals. I can only handle one type right now
        const fkeys = core.Utils.keys(fractionals);
        if (fkeys.length === 1) {
            // Make a note of the factor
            cfact = fkeys[0];
            eq.each((x, index) => {
                if (x.contains(solveFor)) {
                    const parts = explode(x, solveFor);
                    const v = /** @type {NerdamerSymbolType} */ (parts[1]);
                    const p = /** @type {FracType} */ (v.power);
                    if (p.den.gt(1)) {
                        v.power = p.multiply(new core.Frac(cfact));
                        eq.symbols[index] = /** @type {NerdamerSymbolType} */ (_.multiply(v, parts[0]));
                    }
                }
            });
            eq = _.parse(eq);
        }

        // Try for nested sqrts as per issue #486
        addToResult(__.sqrtSolve(eq, solveFor));

        // Polynomial single variable
        if (numvars === 1) {
            if (eq.isPoly(true)) {
                // Try to factor and solve
                const factors = new AlgebraClasses.Factors();

                Factor.factorInner(eq, factors);
                // If the equation has more than one symbolic factor then solve those individually
                if (factors.getNumberSymbolics() > 1) {
                    for (const factorKey in factors.factors) {
                        if (!Object.hasOwn(factors.factors, factorKey)) {
                            continue;
                        }
                        addToResult(solve(factors.factors[factorKey], solveFor));
                    }
                } else {
                    const coeffs = core.Utils.getCoeffs(eq, solveFor);
                    const deg = coeffs.length - 1;
                    let wasCalculated = false;
                    if (vars[0] === solveFor) {
                        // Check to see if all the coefficients are constant
                        if (
                            checkAll(coeffs, coeff => /** @type {NerdamerSymbolType} */ (coeff).group !== core.groups.N)
                        ) {
                            const roots = core.Algebra.proots(eq);
                            // If all the roots are integers then return those
                            if (checkAll(roots, root => !core.Utils.isInt(root))) {
                                // Roots have been calculates
                                wasCalculated = true;
                                roots.forEach(root => {
                                    addToResult(new NerdamerSymbol(root));
                                });
                            }
                        }

                        if (!wasCalculated) {
                            eqns = _.parse(eqns);
                            if (eqns instanceof core.Equation) {
                                eqns = eqns.toLHS();
                            }

                            // We can solve algebraically for degrees 1, 2, 3. The remainder we switch to Jenkins-
                            if (deg === 1) {
                                addToResult(
                                    _.divide(
                                        /** @type {NerdamerSymbolType} */ (coeffs[0]),
                                        /** @type {NerdamerSymbolType} */ (coeffs[1]).negate()
                                    )
                                );
                            } else if (deg === 2) {
                                addToResult(_.expand(__.quad.apply(undefined, coeffs)));
                            } else if (deg === 3) {
                                let cubicSolutions = []; // Set to blank
                                // first try to factor and solve
                                const _factored = Factor.factorInner(/** @type {NerdamerSymbolType} */ (eqns));

                                // If it was successfully factored
                                cubicSolutions = [];
                                if (cubicSolutions.length > 0) {
                                    addToResult(cubicSolutions);
                                } else {
                                    addToResult(__.cubic.apply(undefined, coeffs));
                                }
                            } else {
                                /*
                                 Var sym_roots = csolve(eq, solveFor); 
                                 if(sym_roots.length === 0)
                                 sym_roots = divnconsolve(eq, solveFor);
                                 if(sym_roots.length > 0) 
                                 addToResult(sym_roots);
                                 else
                                 */
                                _A.proots(eq).map(addToResult);
                            }
                        }
                    }
                }
            } else {
                // Attempt Newton
                // Since it's not a polynomial then we'll try to look for a solution using Newton's method
                const hasTrig = eq.hasTrig();
                // We get all the points where a possible zero might exist.
                const points1 = __.getPoints(eq, 0.1);
                const points2 = __.getPoints(eq, 0.05);
                const points3 = __.getPoints(eq, 0.01, true);
                let points = core.Utils.arrayUnique(points1.concat(points2).concat(points3)).sort((a, b) => a - b);
                // Console.log("all points: "+points);
                let i;
                let point;
                let solution;

                // Compile the function
                const f = build(eq.clone());

                // First try to eliminate some points using bisection
                const tPoints = [];
                for (i = 0; i < points.length; i++) {
                    point = points[i];

                    // See if there's a solution at this point
                    solution = __.bisection(point, /** @type {(x: number) => number} */ (f));

                    // If there's no solution then add it to the array for further investigation
                    if (typeof solution === 'undefined') {
                        tPoints.push(point);
                        continue;
                    }

                    // Add the solution to the solution set
                    // console.log("added without Newton: "+solution);
                    // console.log("for: "+eq.text());
                    addToResult(solution, hasTrig);
                }

                // Reset the points to the remaining points
                points = tPoints;
                // Console.log("Newton points: "+points);

                // Build the derivative and compile a function
                const d = _C.diff(eq.clone());
                const fp = build(/** @type {NerdamerSymbolType} */ (d));
                let lastPoint = points[0];
                for (i = 0; i < points.length; i++) {
                    point = points[i];

                    addToResult(
                        __.Newton(
                            point,
                            /** @type {(x: number) => number} */ (f),
                            /** @type {(x: number) => number} */ (fp),
                            lastPoint
                        ),
                        hasTrig
                    );
                    lastPoint = point;
                }

                // Sort by numerical value to be ready for uniquefy filter
                solutions.sort((a, b) => {
                    const sa = a.text('decimals');
                    const sb = b.text('decimals');
                    const xa = Number(sa);
                    const xb = Number(sb);
                    if (isNaN(xa) && isNaN(xb)) {
                        return sa.localeCompare(sb);
                    }
                    if (isNaN(xa) && !isNaN(xb)) {
                        return -1;
                    }
                    if (!isNaN(xa) && isNaN(xb)) {
                        return 1;
                    }
                    return xa - xb;
                });

                // Round to 15 digits
                solutions = solutions.map(a =>
                    a.isConstant() ? new NerdamerSymbol(Number(Number(a).toPrecision(15))) : a
                );

                // Uniquefy to epsilon
                // console.log("solutions: "+solutions);
                solutions = solutions.filter((sol, idx, arr) => {
                    const val = Number(Number(sol).toPrecision(15));
                    const prevVal = Number(arr[idx - 1]);
                    // Console.log("   x: "+val)
                    if (idx === 0 || isNaN(val) || isNaN(prevVal)) {
                        return true;
                    }
                    // If ((Math.abs(val-prevVal) < Settings.EPSILON)) {
                    //     console.log("diff too small: "+val+", "+prevVal);
                    // }
                    return Math.abs(val - prevVal) >= Settings.EPSILON;
                });
                // Console.log("solutions after filter: "+solutions);
            }
            // The idea here is to go through the equation and collect the coefficients
            // place them in an array and call the quad or cubic function to get the results
        } else if (!eq.hasFunc(solveFor) && eq.isComposite()) {
            try {
                // This is where solving certain quads goes wrong

                const factored = Factor.factorInner(eq.clone());
                const test = _.expand(/** @type {NerdamerSymbolType} */ (_.parse(factored)));
                const test2 = _.expand(eq.clone());
                const diff = /** @type {NerdamerSymbolType} */ (_.subtract(test, test2));
                let validFactorization = true;
                if (!diff.equals(0)) {
                    // Console.log("factored: "+test);
                    // console.log("original: "+test2);
                    validFactorization = false;
                }

                if (validFactorization && factored.group === CB) {
                    factored.each(factor => {
                        addToResult(solve(factor, solveFor));
                    });
                } else {
                    const coeffs = core.Utils.getCoeffs(eq, solveFor);

                    const l = coeffs.length;
                    const deg = l - 1; // The degree of the polynomial
                    // get the denominator and make sure it doesn't have x

                    // handle the problem based on the degree
                    switch (deg) {
                        case 0: {
                            const separated = separate(eq);
                            const lhs = separated[0];
                            const rhs = separated[1];

                            if (lhs.group === core.groups.EX) {
                                // We have a*b^(mx) = rhs
                                // => log(b^(mx)) = log(rhs/a)
                                // => mx*log(b) = log(rhs/a)
                                // => x = log(rhs/a)/(m*log(b))

                                const log = core.Settings.LOG;
                                const exprStr = `${log}((${rhs})/(${lhs.multiplier}))/(${log}(${lhs.value})*${/** @type {NerdamerSymbolType} */ (lhs.power).multiplier})`;
                                const parsed = _.parse(exprStr);
                                addToResult(parsed);
                            }
                            break;
                        }
                        case 1:
                            // Nothing to do but to return the quotient of the constant and the LT
                            // e.g. 2*x-1
                            addToResult(
                                _.divide(
                                    /** @type {NerdamerSymbolType} */ (coeffs[0]),
                                    /** @type {NerdamerSymbolType} */ (coeffs[1]).negate()
                                )
                            );
                            break;
                        case 2:
                            addToResult(__.quad.apply(undefined, coeffs));
                            break;
                        case 3:
                            addToResult(__.cubic.apply(undefined, coeffs));
                            break;
                        case 4:
                            addToResult(__.quartic.apply(undefined, coeffs));
                            break;
                        default:
                            addToResult(__.csolve(eq, solveFor));
                            if (solutions.length === 0) {
                                addToResult(__.divideAndConquer(eq, solveFor));
                            }
                    }

                    if (solutions.length === 0) {
                        // Try factoring
                        addToResult(solve(factored, solveFor, solutions, depth));
                    }
                }
            } catch (e) {
                /* Something went wrong. EXITING*/
                if (e.message === 'timeout') {
                    throw e;
                }
            }
        } else {
            try {
                const rw = __.rewrite(eq, null, solveFor);
                const lhs = rw[0];
                let rhs = rw[1];
                if (lhs.group === FN) {
                    if (lhs.fname === 'abs') {
                        // Solve only if solveFor was the only arg
                        if (lhs.args[0].toString() === solveFor) {
                            addToResult([rhs.clone(), rhs.negate()]);
                        }
                    } else if (lhs.fname === 'sin') {
                        // Asin
                        addToResult(__.inverseFunctionSolve('asin', lhs, rhs));
                    } else if (lhs.fname === 'cos') {
                        // Asin
                        addToResult(__.inverseFunctionSolve('acos', lhs, rhs));
                    } else if (lhs.fname === 'tan') {
                        // Asin
                        addToResult(__.inverseFunctionSolve('atan', lhs, rhs));
                    } else if (lhs.fname === core.Settings.LOG) {
                        // Ax+b comes back as [a, x, ax, b];
                        const parts = explode(lhs.args[0], solveFor);
                        // Check if x is by itself
                        const x = /** @type {NerdamerSymbolType} */ (parts[1]);
                        if (x.group === S) {
                            rhs = _.divide(
                                _.subtract(
                                    _.pow(
                                        lhs.args.length > 1 ? lhs.args[1] : new NerdamerSymbol('e'),
                                        _.divide(rhs, _.parse(lhs.multiplier))
                                    ),
                                    parts[3]
                                ),
                                parts[0]
                            );
                            const newEq = new Equation(x, rhs).toLHS();
                            addToResult(solve(newEq, solveFor));
                        }
                    } else {
                        addToResult(_.subtract(lhs, rhs));
                    }
                } else {
                    const neq = new Equation(lhs, rhs).toLHS(); // Create a new equation

                    if (neq.equals(eq)) {
                        throw new Error('Stopping. No stop condition exists');
                    }
                    addToResult(solve(neq, solveFor));
                }
            } catch (error) {
                if (error.message === 'timeout') {
                    throw error;
                }
                // Let's try this another way
                // 1. if the symbol is in the form a*b*c*... then the solution is zero if
                // either a or b or c is zero.
                if (eq.group === CB) {
                    addToResult(0);
                } else if (eq.group === CP) {
                    const separated = separate(eq);
                    const lhs = separated[0];
                    const rhs = separated[1];

                    // Reduce the equation
                    if (lhs.group === core.groups.EX && lhs.value === solveFor) {
                        // Change the base of both sides
                        const p = /** @type {NerdamerSymbolType} */ (lhs.power.clone().invert());
                        addToResult(_.pow(rhs, p));
                    }
                }
            }
        }

        if (cfact) {
            solutions = solutions.map(sol => _.pow(sol, new NerdamerSymbol(cfact)));
        }

        // Perform some cleanup but don't do it agains arrays, etc
        // Check it actually evaluates to zero
        if (isSymbol(eqns)) {
            /** @type {Record<string, NerdamerSymbolType>} */
            const knowns = {};
            solutions = solutions.filter(sol => {
                try {
                    knowns[solveFor] = sol;
                    const zero = Number(evaluate(eqns, knowns));

                    // Allow symbolic answers
                    if (isNaN(zero)) {
                        return true;
                    }
                    return true;
                } catch (e) {
                    if (e.message === 'timeout') {
                        throw e;
                    }
                    return false;
                }
            });
        }

        return solutions;
    }

    // Register the functions for external use
    nerdamer.register([
        {
            name: 'solveEquations',
            parent: 'nerdamer',
            numargs: -1,
            visible: true,
            build() {
                return solve; // Comment out to return a vector
                /*
                 return function() {
                 return core.Utils.convertToVector(solve.apply(null, arguments));
                 };
                 */
            },
        },
        {
            name: 'solve',
            parent: 'Solve',
            numargs: 2,
            visible: true,
            /** @returns {(...args: unknown[]) => unknown} */
            build() {
                return /** @type {(...args: unknown[]) => unknown} */ (core.Solve.solve);
            },
        },
        {
            name: 'setEquation',
            parent: 'Solve',
            numargs: 2,
            visible: true,
            build() {
                return setEq;
            },
        },
    ]);
    nerdamer.updateAPI();
})();
