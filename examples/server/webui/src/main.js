import './styles.scss';
import { createApp, defineComponent, shallowRef, computed, h } from 'vue/dist/vue.esm-bundler.js';
import MarkdownIt from 'markdown-it';
import TextLineStream from 'textlinestream';

// math formula rendering
import 'katex/dist/katex.min.css';
import markdownItKatexGpt from './katex-gpt';
import markdownItKatexNormal from '@vscode/markdown-it-katex';

// code highlighting
import hljs from './highlight-config';
import daisyuiThemes from 'daisyui/src/theming/themes';

// ponyfill for missing ReadableStream asyncIterator on Safari
import { asyncIterator } from '@sec-ant/readable-stream/ponyfill/asyncIterator';

const isDev = import.meta.env.MODE === 'development';

// utility functions
const isString = (x) => !!x.toLowerCase;
const isBoolean = (x) => x === true || x === false;
const isNumeric = (n) => !isString(n) && !isNaN(n) && !isBoolean(n);
const escapeAttr = (str) => str.replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const copyStr = (textToCopy) => {
  // Navigator clipboard api needs a secure context (https)
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(textToCopy);
  } else {
    // Use the 'out of viewport hidden text area' trick
    const textArea = document.createElement('textarea');
    textArea.value = textToCopy;
    // Move textarea out of the viewport so it's not visible
    textArea.style.position = 'absolute';
    textArea.style.left = '-999999px';
    document.body.prepend(textArea);
    textArea.select();
    document.execCommand('copy');
  }
};

// constants
const BASE_URL = isDev
  ? (localStorage.getItem('base') || 'https://localhost:8080') // for debugging
  : (new URL('.', document.baseURI).href).toString().replace(/\/$/, ''); // for production
console.log({ BASE_URL });

const CONFIG_DEFAULT = {
  // Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value. Do not use null or undefined for default value.
  apiKey: '',
  systemMessage: 'You are a helpful assistant.',
  showTokensPerSecond: false,
  // make sure these default values are in sync with `common.h`
  samplers: 'edkypmxt',
  temperature: 0.8,
  dynatemp_range: 0.0,
  dynatemp_exponent: 1.0,
  top_k: 40,
  top_p: 0.95,
  min_p: 0.05,
  xtc_probability: 0.0,
  xtc_threshold: 0.1,
  typical_p: 1.0,
  repeat_last_n: 64,
  repeat_penalty: 1.0,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  dry_multiplier: 0.0,
  dry_base: 1.75,
  dry_allowed_length: 2,
  dry_penalty_last_n: -1,
  max_tokens: -1,
  custom: '', // custom json-stringified object
};
const CONFIG_INFO = {
  apiKey: 'Set the API Key if you are using --api-key option for the server.',
  systemMessage: 'The starting message that defines how model should behave.',
  samplers: 'The order at which samplers are applied, in simplified way. Default is "dkypmxt": dry->top_k->typ_p->top_p->min_p->xtc->temperature',
  temperature: 'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
  dynatemp_range: 'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
  dynatemp_exponent: 'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
  top_k: 'Keeps only k top tokens.',
  top_p: 'Limits tokens to those that together have a cumulative probability of at least p',
  min_p: 'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
  xtc_probability: 'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
  xtc_threshold: 'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
  typical_p: 'Sorts and limits tokens based on the difference between log-probability and entropy.',
  repeat_last_n: 'Last n tokens to consider for penalizing repetition',
  repeat_penalty: 'Controls the repetition of token sequences in the generated text',
  presence_penalty: 'Limits tokens based on whether they appear in the output or not.',
  frequency_penalty: 'Limits tokens based on how often they appear in the output.',
  dry_multiplier: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
  dry_base: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
  dry_allowed_length: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
  dry_penalty_last_n: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
  max_tokens: 'The maximum number of token per output.',
  custom: '', // custom json-stringified object
};
// config keys having numeric value (i.e. temperature, top_k, top_p, etc)
const CONFIG_NUMERIC_KEYS = Object.entries(CONFIG_DEFAULT).filter(e => isNumeric(e[1])).map(e => e[0]);
// list of themes supported by daisyui
const THEMES = ['light', 'dark']
  // make sure light & dark are always at the beginning
  .concat(Object.keys(daisyuiThemes).filter(t => t !== 'light' && t !== 'dark'));

// markdown support
const VueMarkdown = defineComponent(
  (props) => {
    const md = shallowRef(new MarkdownIt({
      breaks: true,
      highlight: function (str, lang) { // Add highlight.js
        if (lang && hljs.getLanguage(lang)) {
          try {
            return '<pre dir="auto"><code class="hljs">' +
                   hljs.highlight(str, { language: lang, ignoreIllegals: true }).value +
                   '</code></pre>';
          } catch (__) {}
        }
        return '<pre dir="auto"><code class="hljs">' + md.value.utils.escapeHtml(str) + '</code></pre>';
      }
    }));
    // support latex with double dollar sign and square brackets
    md.value.use(markdownItKatexGpt, {
      delimiters: [
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$$', right: '$$', display: false },
        // do not add single dollar sign here, other wise it will confused with dollar used for money symbol
      ],
      throwOnError: false,
    });
    // support latex with single dollar sign
    md.value.use(markdownItKatexNormal, { throwOnError: false });
    // add copy button to code blocks
    const origFenchRenderer = md.value.renderer.rules.fence;
    md.value.renderer.rules.fence = (tokens, idx, ...args) => {
      const content = tokens[idx].content;
      const origRendered = origFenchRenderer(tokens, idx, ...args);
      return `<div class="relative my-4">
        <div class="text-right sticky top-4 mb-2 mr-2 h-0">
          <button class="badge btn-mini" onclick="copyStr(${escapeAttr(JSON.stringify(content))})">📋 Copy</button>
        </div>
        ${origRendered}
      </div>`;
    };
    window.copyStr = copyStr;
    const content = computed(() => md.value.render(props.source));
    return () => h('div', { innerHTML: content.value });
  },
  { props: ['source'] }
);

// input field to be used by settings modal
const SettingsModalShortInput = defineComponent({
  template: document.getElementById('settings-modal-short-input').innerHTML,
  props: {
    label: { type: String, required: false },
    configKey: String,
    configDefault: Object,
    configInfo: Object,
    modelValue: [Object, String, Number],
  },
});

// message bubble component
const MessageBubble = defineComponent({
  components: {
    VueMarkdown
  },
  template: document.getElementById('message-bubble').innerHTML,
  props: {
    config: Object,
    msg: Object,
    isGenerating: Boolean,
    editUserMsgAndRegenerate: Function,
    regenerateMsg: Function,
  },
  data() {
    return {
      editingContent: null,
    };
  },
  computed: {
    timings() {
      if (!this.msg.timings) return null;
      return {
        ...this.msg.timings,
        prompt_per_second: this.msg.timings.prompt_n / (this.msg.timings.prompt_ms / 1000),
        predicted_per_second: this.msg.timings.predicted_n / (this.msg.timings.predicted_ms / 1000),
      };
    }
  },
  methods: {
    copyMsg() {
      copyStr(this.msg.content);
    },
    editMsg() {
      this.editUserMsgAndRegenerate({
        ...this.msg,
        content: this.editingContent,
      });
      this.editingContent = null;
    },
  },
});

// coversations is stored in localStorage
// format: { [convId]: { id: string, lastModified: number, messages: [...] } }
// convId is a string prefixed with 'conv-'
const StorageUtils = {
  // manage conversations
  getAllConversations() {
    const res = [];
    for (const key in localStorage) {
      if (key.startsWith('conv-')) {
        res.push(JSON.parse(localStorage.getItem(key)));
      }
    }
    res.sort((a, b) => b.lastModified - a.lastModified);
    return res;
  },
  // can return null if convId does not exist
  getOneConversation(convId) {
    return JSON.parse(localStorage.getItem(convId) || 'null');
  },
  // if convId does not exist, create one
  appendMsg(convId, msg) {
    if (msg.content === null) return;
    const conv = StorageUtils.getOneConversation(convId) || {
      id: convId,
      lastModified: Date.now(),
      messages: [],
    };
    conv.messages.push(msg);
    conv.lastModified = Date.now();
    localStorage.setItem(convId, JSON.stringify(conv));
  },
  getNewConvId() {
    return `conv-${Date.now()}`;
  },
  remove(convId) {
    localStorage.removeItem(convId);
  },
  filterAndKeepMsgs(convId, predicate) {
    const conv = StorageUtils.getOneConversation(convId);
    if (!conv) return;
    conv.messages = conv.messages.filter(predicate);
    conv.lastModified = Date.now();
    localStorage.setItem(convId, JSON.stringify(conv));
  },
  popMsg(convId) {
    const conv = StorageUtils.getOneConversation(convId);
    if (!conv) return;
    const msg = conv.messages.pop();
    conv.lastModified = Date.now();
    if (conv.messages.length === 0) {
      StorageUtils.remove(convId);
    } else {
      localStorage.setItem(convId, JSON.stringify(conv));
    }
    return msg;
  },

  // manage config
  getConfig() {
    const savedVal = JSON.parse(localStorage.getItem('config') || '{}');
    // to prevent breaking changes in the future, we always provide default value for missing keys
    return {
      ...CONFIG_DEFAULT,
      ...savedVal,
    };
  },
  setConfig(config) {
    localStorage.setItem('config', JSON.stringify(config));
  },
  getTheme() {
    return localStorage.getItem('theme') || 'auto';
  },
  setTheme(theme) {
    if (theme === 'auto') {
      localStorage.removeItem('theme');
    } else {
      localStorage.setItem('theme', theme);
    }
  },
};

// scroll to bottom of chat messages
// if requiresNearBottom is true, only auto-scroll if user is near bottom
const chatScrollToBottom = (requiresNearBottom) => {
  const msgListElem = document.getElementById('messages-list');
  const spaceToBottom = msgListElem.scrollHeight - msgListElem.scrollTop - msgListElem.clientHeight;
  if (!requiresNearBottom || (spaceToBottom < 100)) {
    setTimeout(() => msgListElem.scrollTo({ top: msgListElem.scrollHeight }), 1);
  }
};

// wrapper for SSE
async function* sendSSEPostRequest(url, fetchOptions) {
  const res = await fetch(url, fetchOptions);
  const lines = res.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TextLineStream());
  for await (const line of asyncIterator(lines)) {
    if (isDev) console.log({line});
    if (line.startsWith('data:') && !line.endsWith('[DONE]')) {
      const data = JSON.parse(line.slice(5));
      yield data;
    } else if (line.startsWith('error:')) {
      const data = JSON.parse(line.slice(6));
      throw new Error(data.message || 'Unknown error');
    }
  }
};

const mainApp = createApp({
  components: {
    VueMarkdown,
    SettingsModalShortInput,
    MessageBubble,
  },
  data() {
    return {
      conversations: StorageUtils.getAllConversations(),
      messages: [], // { id: number, role: 'user' | 'assistant', content: string }
      viewingConvId: StorageUtils.getNewConvId(),
      inputMsg: '',
      isGenerating: false,
      pendingMsg: null, // the on-going message from assistant
      stopGeneration: () => {},
      selectedTheme: StorageUtils.getTheme(),
      config: StorageUtils.getConfig(),
      showConfigDialog: false,
      // const
      themes: THEMES,
      configDefault: {...CONFIG_DEFAULT},
      configInfo: {...CONFIG_INFO},
      isDev,
    }
  },
  computed: {},
  mounted() {
    document.getElementById('app').classList.remove('opacity-0'); // show app
    // scroll to the bottom when the pending message height is updated
    const pendingMsgElem = document.getElementById('pending-msg');
    const resizeObserver = new ResizeObserver(() => {
      if (this.isGenerating) chatScrollToBottom(true);
    });
    resizeObserver.observe(pendingMsgElem);
    this.setSelectedTheme(this.selectedTheme);
  },
  watch: {
    viewingConvId: function(val, oldVal) {
      if (val != oldVal) {
        this.fetchMessages();
        chatScrollToBottom();
        this.hideSidebar();
      }
    }
  },
  methods: {
    hideSidebar() {
      document.getElementById('toggle-drawer').checked = false;
    },
    setSelectedTheme(theme) {
      this.selectedTheme = theme;
      document.body.setAttribute('data-theme', theme);
      document.body.setAttribute('data-color-scheme', daisyuiThemes[theme]?.['color-scheme'] ?? 'auto');
      StorageUtils.setTheme(theme);
    },
    newConversation() {
      if (this.isGenerating) return;
      this.viewingConvId = StorageUtils.getNewConvId();
    },
    setViewingConv(convId) {
      if (this.isGenerating) return;
      this.viewingConvId = convId;
    },
    deleteConv(convId) {
      if (this.isGenerating) return;
      if (window.confirm('Are you sure to delete this conversation?')) {
        StorageUtils.remove(convId);
        if (this.viewingConvId === convId) {
          this.viewingConvId = StorageUtils.getNewConvId();
        }
        this.fetchConversation();
        this.fetchMessages();
      }
    },
    downloadConv(convId) {
      const conversation = StorageUtils.getOneConversation(convId);
      if (!conversation) {
        alert('Conversation not found.');
        return;
      }
      const conversationJson = JSON.stringify(conversation, null, 2);
      const blob = new Blob([conversationJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation_${convId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
    async sendMessage() {
      if (!this.inputMsg) return;
      const currConvId = this.viewingConvId;

      StorageUtils.appendMsg(currConvId, {
        id: Date.now(),
        role: 'user',
        content: this.inputMsg,
      });
      this.fetchConversation();
      this.fetchMessages();
      this.inputMsg = '';
      this.generateMessage(currConvId);
      chatScrollToBottom();
    },
    async generateMessage(currConvId) {
      if (this.isGenerating) return;
      this.pendingMsg = { id: Date.now()+1, role: 'assistant', content: null };
      this.isGenerating = true;

      try {
        const abortController = new AbortController();
        this.stopGeneration = () => abortController.abort();
        const params = {
          messages: [
            { role: 'system', content: this.config.systemMessage },
            ...this.messages,
          ],
          stream: true,
          cache_prompt: true,
          samplers: this.config.samplers,
          temperature: this.config.temperature,
          dynatemp_range: this.config.dynatemp_range,
          dynatemp_exponent: this.config.dynatemp_exponent,
          top_k: this.config.top_k,
          top_p: this.config.top_p,
          min_p: this.config.min_p,
          typical_p: this.config.typical_p,
          xtc_probability: this.config.xtc_probability,
          xtc_threshold: this.config.xtc_threshold,
          repeat_last_n: this.config.repeat_last_n,
          repeat_penalty: this.config.repeat_penalty,
          presence_penalty: this.config.presence_penalty,
          frequency_penalty: this.config.frequency_penalty,
          dry_multiplier: this.config.dry_multiplier,
          dry_base: this.config.dry_base,
          dry_allowed_length: this.config.dry_allowed_length,
          dry_penalty_last_n: this.config.dry_penalty_last_n,
          max_tokens: this.config.max_tokens,
          timings_per_token: !!this.config.showTokensPerSecond,
          ...(this.config.custom.length ? JSON.parse(this.config.custom) : {}),
        };
        const chunks = sendSSEPostRequest(`${BASE_URL}/v1/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(this.config.apiKey ? {'Authorization': `Bearer ${this.config.apiKey}`} : {})
          },
          body: JSON.stringify(params),
          signal: abortController.signal,
        });
        for await (const chunk of chunks) {
          const stop = chunk.stop;
          const addedContent = chunk.choices[0].delta.content;
          const lastContent = this.pendingMsg.content || '';
          if (addedContent) {
            this.pendingMsg = {
              id: this.pendingMsg.id,
              role: 'assistant',
              content: lastContent + addedContent,
            };
          }
          const timings = chunk.timings;
          if (timings && this.config.showTokensPerSecond) {
            // only extract what's really needed, to save some space
            this.pendingMsg.timings = {
              prompt_n: timings.prompt_n,
              prompt_ms: timings.prompt_ms,
              predicted_n: timings.predicted_n,
              predicted_ms: timings.predicted_ms,
            };
          }
        }

        StorageUtils.appendMsg(currConvId, this.pendingMsg);
        this.fetchConversation();
        this.fetchMessages();
        setTimeout(() => document.getElementById('msg-input').focus(), 1);
      } catch (error) {
        if (error.name === 'AbortError') {
          // user stopped the generation via stopGeneration() function
          StorageUtils.appendMsg(currConvId, this.pendingMsg);
          this.fetchConversation();
          this.fetchMessages();
        } else {
          console.error(error);
          alert(error);
          // pop last user message
          const lastUserMsg = StorageUtils.popMsg(currConvId);
          this.inputMsg = lastUserMsg ? lastUserMsg.content : '';
        }
      }

      this.pendingMsg = null;
      this.isGenerating = false;
      this.stopGeneration = () => {};
      this.fetchMessages();
      chatScrollToBottom();
    },

    // message actions
    regenerateMsg(msg) {
      if (this.isGenerating) return;
      // TODO: somehow keep old history (like how ChatGPT has different "tree"). This can be done by adding "sub-conversations" with "subconv-" prefix, and new message will have a list of subconvIds
      const currConvId = this.viewingConvId;
      StorageUtils.filterAndKeepMsgs(currConvId, (m) => m.id < msg.id);
      this.fetchConversation();
      this.fetchMessages();
      this.generateMessage(currConvId);
    },
    editUserMsgAndRegenerate(msg) {
      if (this.isGenerating) return;
      const currConvId = this.viewingConvId;
      const newContent = msg.content;
      StorageUtils.filterAndKeepMsgs(currConvId, (m) => m.id < msg.id);
      StorageUtils.appendMsg(currConvId, {
        id: Date.now(),
        role: 'user',
        content: newContent,
      });
      this.fetchConversation();
      this.fetchMessages();
      this.generateMessage(currConvId);
    },

    // settings dialog methods
    closeAndSaveConfigDialog() {
      try {
        if (this.config.custom.length) JSON.parse(this.config.custom);
      } catch (error) {
        alert('Invalid JSON for custom config. Please either fix it or leave it empty.');
        return;
      }
      for (const key of CONFIG_NUMERIC_KEYS) {
        if (isNaN(this.config[key]) || this.config[key].toString().trim().length === 0) {
          alert(`Invalid number for ${key} (expected an integer or a float)`);
          return;
        }
        this.config[key] = parseFloat(this.config[key]);
      }
      this.showConfigDialog = false;
      StorageUtils.setConfig(this.config);
    },
    closeAndDiscardConfigDialog() {
      this.showConfigDialog = false;
      this.config = StorageUtils.getConfig();
    },
    resetConfigDialog() {
      if (window.confirm('Are you sure to reset all settings?')) {
        this.config = {...CONFIG_DEFAULT};
      }
    },

    // sync state functions
    fetchConversation() {
      this.conversations = StorageUtils.getAllConversations();
    },
    fetchMessages() {
      this.messages = StorageUtils.getOneConversation(this.viewingConvId)?.messages ?? [];
    },

    // debug functions
    async debugImportDemoConv() {
      const res = await fetch('/demo-conversation.json');
      const demoConv = await res.json();
      StorageUtils.remove(demoConv.id);
      for (const msg of demoConv.messages) {
        StorageUtils.appendMsg(demoConv.id, msg);
      }
      this.fetchConversation();
    }
  },
});
mainApp.config.errorHandler = alert;
try {
  mainApp.mount('#app');
} catch (err) {
  console.error(err);
  document.getElementById('app').innerHTML = `<div style="margin:2em auto">
    Failed to start app. Please try clearing localStorage and try again.<br/>
    <br/>
    <button class="btn" onClick="localStorage.clear(); window.location.reload();">Clear localStorage</button>
  </div>`;
}
