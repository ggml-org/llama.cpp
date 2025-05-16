export const BASE_URL = new URL('.', document.baseURI).href
  .toString()
  .replace(/\/$/, '');

export type AppConfig = {
  // Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value. Do not use null or undefined for default value.
  // Do not use nested objects, keep it single level. Prefix the key if you need to group them.
  apiKey: string;
  systemMessage: string;
  showTokensPerSecond: boolean;
  showThoughtInProgress: boolean;
  excludeThoughtOnReq: boolean;
  pasteLongTextToFileLen: number;
  pdfAsImage: boolean;

  // make sure these default values are in sync with `common.h`
  samplers: string;
  temperature: number;
  dynatemp_range: number;
  dynatemp_exponent: number;
  top_k: number;
  top_p: number;
  min_p: number;
  xtc_probability: number;
  xtc_threshold: number;
  typical_p: number;
  repeat_last_n: number;
  repeat_penalty: number;
  presence_penalty: number;
  frequency_penalty: number;
  dry_multiplier: number;
  dry_base: number;
  dry_allowed_length: number;
  dry_penalty_last_n: number;
  max_tokens: number;
  custom: string; // custom json-stringified object

  // experimental features
  pyIntepreterEnabled: boolean;
};

export const CONFIG_INFO: Record<string, string> = {
  apiKey: 'Set the API Key if you are using --api-key option for the server.',
  systemMessage: 'The starting message that defines how model should behave.',
  pasteLongTextToFileLen:
    'On pasting long text, it will be converted to a file. You can control the file length by setting the value of this parameter. Value 0 means disable.',
  samplers:
    'The order at which samplers are applied, in simplified way. Default is "dkypmxt": dry->top_k->typ_p->top_p->min_p->xtc->temperature',
  temperature:
    'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
  dynatemp_range:
    'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
  dynatemp_exponent:
    'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
  top_k: 'Keeps only k top tokens.',
  top_p:
    'Limits tokens to those that together have a cumulative probability of at least p',
  min_p:
    'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
  xtc_probability:
    'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
  xtc_threshold:
    'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
  typical_p:
    'Sorts and limits tokens based on the difference between log-probability and entropy.',
  repeat_last_n: 'Last n tokens to consider for penalizing repetition',
  repeat_penalty:
    'Controls the repetition of token sequences in the generated text',
  presence_penalty:
    'Limits tokens based on whether they appear in the output or not.',
  frequency_penalty:
    'Limits tokens based on how often they appear in the output.',
  dry_multiplier:
    'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
  dry_base:
    'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
  dry_allowed_length:
    'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
  dry_penalty_last_n:
    'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
  max_tokens: 'The maximum number of token per output.',
  custom: '', // custom json-stringified object
};

import daisyuiThemes from 'daisyui/theme/object';

export const THEMES = ['light', 'dark'].concat(
  Object.keys(daisyuiThemes).filter((t) => t !== 'light' && t !== 'dark')
);

export const isDev = import.meta.env.MODE === 'development';
