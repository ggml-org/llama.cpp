/*
 * Author : Martin Donk
 * Website : http://www.nerdamer.com
 * Email : martin.r.donk@gmail.com
 * Source : https://github.com/jiggzson/nerdamer
 * Can be used to load all add-ons with one require
 */

const nerdamer = require('./nerdamer.core.js');
require('./Algebra.js');
require('./Calculus.js');
require('./Solve.js');
require('./Extra.js');

// Export nerdamer
module.exports = nerdamer;
