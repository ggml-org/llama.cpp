
# SimpleChat / AnveshikaSallap

by Humans for All.

## quickstart

To run from the build dir

bin/llama-server -m path/model.gguf --path ../tools/server/public_simplechat --jinja

Continue reading for the details.

## overview

This simple web frontend, allows triggering/testing the server's /completions or /chat/completions endpoints
in a simple way with minimal code from a common code base. Additionally it also allows end users to have
single or multiple independent chat sessions with back and forth chatting to an extent, with the ai llm model
at a basic level, with their own system prompts.

This allows seeing the generated text / ai-model response in oneshot at the end, after it is fully generated,
or potentially as it is being generated, in a streamed manner from the server/ai-model.

![Chat and Settings (old) screens](./simplechat_screens.webp "Chat and Settings (old) screens")

Auto saves the chat session locally as and when the chat is progressing and inturn at a later time when you
open SimpleChat, option is provided to restore the old chat session, if a matching one exists. In turn if
any of those chat sessions were pending wrt user triggering a tool call or submitting a tool call response,
the ui is setup as needed for end user to continue with those previously saved sessions, from where they
left off.

The UI follows a responsive web design so that the layout can adapt to available display space in a usable
enough manner, in general.

Allows developer/end-user to control some of the behaviour by updating gMe members from browser's devel-tool
console. Parallely some of the directly useful to end-user settings can also be changed using the provided
settings ui. Each chat session has its own set of config entries and thus its own setting, this allows one
to have independent chat sessions with different instances of llama-server and or with different configs.

For GenAi/LLM models supporting tool / function calling, allows one to interact with them and explore use of
ai driven augmenting of the knowledge used for generating answers as well as for cross checking ai generated
answers logically / programatically and by checking with other sources and lot more by making using of the
simple yet useful predefined tools / functions provided by this chat client. The end user is provided full
control over tool calling and response submitting.

For GenAi/LLM models which support reasoning, the thinking of the model will be shown to the end user as the
model is running through its reasoning.

For GenAi/LLM models with vision support, one can specify image file and get the ai to respond wrt the same.

NOTE: As all genai/llm web service apis may or may not expose the model context length directly, and also
as using ai out of band for additional parallel work may not be efficient given the loading of current systems
by genai/llm models, so client logic doesnt provide any adaptive culling of old messages nor of replacing them
with summary of their content etal. However there is a optional sliding window based chat logic, which provides
a simple minded culling of old messages from the chat history before sending to the ai model.

NOTE: Wrt options sent with the request, it mainly sets temperature, max_tokens and optionaly stream as well
as tool_calls mainly for now. However if someone wants they can update the js file or equivalent member in
gMe as needed.

NOTE: One may be able to use this to chat with openai api web-service /chat/completions endpoint, in a limited
/ minimal way. One will need to set model, openai url and authorization bearer key in settings ui.


## usage

One could run this web frontend directly using server itself or if anyone is thinking of adding a built in web
frontend to configure the server over http(s) or so, then run this web frontend using something like python's
http module.

### running directly using tools/server

./llama-server -m path/model.gguf --path tools/server/public_simplechat [--port PORT --jinja]

### running using python3's server module

first run tools/server
* ./llama-server -m path/model.gguf [--port PORT --jinja]

next run this web front end in tools/server/public_simplechat
* cd ../tools/server/public_simplechat
* python3 -m http.server PORT

### for tool calling

remember to

* pass --jinja to llama-server to enable tool calling support from the llama server ai engine end.

* tools.enabled needs to be true in the settings page of a chat session, in the client side gui.

* use a GenAi/LLM model which supports tool calling.

* if fetch web page, web search, pdf-to-text, ... tool call is needed remember to run bundled
  local.tools/simplemcp.py
  helper along with its config file, before using/loading this client ui through a browser

  * cd tools/server/public_simplechat/local.tools; python3 ./simplemcp.py --config simplemcp.json

  * remember that this is a relatively minimal dumb mcp(ish) server logic with few builtin tool calls
  related to fetching raw html or stripped plain text equivalent or pdf text content.
  Be careful when accessing web through this and use it only with known safe sites.

  * look into local.tools/simplemcp.json for specifying

    * the white list of acl.schemes
      * you may want to use this to disable local file access and or disable http access,
        and inturn retaining only https based urls or so.
    * the white list of acl.domains
      * review and update this to match your needs.
    * the shared bearer token between simplemcp server and chat client
    * the public certificate and private key files to enable https mode
      * sec.certFile and sec.keyFile

* other builtin tool / function calls like datetime, calculator, javascript runner, DataStore,
  external ai dont require this bundled simplemcp.py helper.

### for vision models

* remember to also specify the multimodal related gguf file directly using -mmproj or by using -hf to
  fetch the llm model and its mmproj gguf from huggingface.
* additionally you may need to specify a large enough -batch-size (ex 8k) and -ubatch-size (ex 2k)


### using the front end

Open this simple web front end from your local browser

* http://127.0.0.1:PORT/index.html

Once inside

* If you want to, you can change many of the default chat session settings
  * the base url (ie ip addr / domain name, port)
  * chat (default) vs completion mode
  * try trim garbage in response or not
  * amount of chat history in the context sent to server/ai-model
  * oneshot or streamed mode.
  * use built in tool calling or not and its related params.
  * ...

* In completion mode >> note: most recent work has been in chat mode <<
  * one normally doesnt use a system prompt in completion mode.
  * logic by default doesnt insert any role specific "ROLE: " prefix wrt each role's message.
    If the model requires any prefix wrt user role messages, then the end user has to
    explicitly add the needed prefix, when they enter their chat message.
    Similarly if the model requires any prefix to trigger assistant/ai-model response,
    then the end user needs to enter the same.
    This keeps the logic simple, while still giving flexibility to the end user to
    manage any templating/tagging requirement wrt their messages to the model.
  * the logic doesnt insert newline at the begining and end wrt the prompt message generated.
    However if the chat being sent to /completions end point has more than one role's message,
    then insert newline when moving from one role's message to the next role's message, so
    that it can be clearly identified/distinguished.
  * given that /completions endpoint normally doesnt add additional chat-templating of its
    own, the above ensures that end user can create a custom single/multi message combo with
    any tags/special-tokens related chat templating to test out model handshake. Or enduser
    can use it just for normal completion related/based query.

* If you want to provide a system prompt, then ideally enter it first, before entering any user query.
  Normally Completion mode doesnt need system prompt, while Chat mode can generate better/interesting
  responses with a suitable system prompt.
  * one can change the system prompt any time during chat, by changing the contents of system prompt.
  * inturn the updated/changed system prompt will be inserted into the chat session.
  * this allows for the subsequent user chatting to be driven by the new system prompt set above.
  * devel note: chat.add_system_anytime and related

* Enter your query and either press enter or click on the submit button.
  * If you want to insert enter (\n) as part of your chat/query to ai model, use shift+enter.
  * If the tool response has been placed into user input textarea, its color is changed to help user
    identify the same easily.
  * allow user to specify image files, for vision models.

* Wait for the logic to communicate with the server and get the response.
  * the user is not allowed to enter any fresh query during this time.
  * the user input box will be disabled and a working message will be shown in it.
  * if trim garbage is enabled, logic will try to trim repeating text kind of garbage to some extent.

* any reasoning / thinking by the model is shown to the end user, as it is occuring, if the ai model
  shares the same over the http interface.

* tool calling flow when working with ai models which support tool / function calling
  * if tool calling is enabled and the user query results in need for one of the builtin tools to be
    called, then the ai response might include request for tool call.
  * the SimpleChat client will show details of the tool call (ie tool name and args passed) requested
    and allow the user to trigger it as is or after modifying things as needed.
    NOTE: Tool sees the original tool call only, for now
  * inturn returned / generated result is placed into user query entry text area,
    and the color of the user query text area is changed to indicate the same.
  * if user is ok with the tool response, they can click submit to send the same to the GenAi/LLM.
    User can even modify the response generated by the tool, if required, before submitting.
  * ALERT: Sometimes the reasoning or chat from ai model may indicate tool call, but you may actually
    not get/see a tool call, in such situations, dont forget to cross check that tool calling is
    enabled in the settings. Also click on the current chat session's button at the top, to refresh
    the ui, just in case.

* when the user is going through the chat messages in the chat session, they can
  * delete any message from the chat session,
    * remember that you need to maintain the expected sequence of chat message roles
      ie user - assistant - {tool response} - user - assistant - kind of sequence.
  * copy text content of messages to clipboard.

* {spiral}ClearCurrentChat/Refresh
  * use the clear button to clear the currently active chat session.
  * just refresh the page, to reset wrt the chat history and system prompts across chat sessions
    and start afresh.
    * This also helps if you had forgotten to start the bundled simplemcp.py server before hand.
      Start the simplemcp.py server and refresh the client ui page, to get access to web access
      related tool calls.
      * starting new chat session, after starting simplemcp, will also give access to tool calls
        exposed by simplemcp, in that new chat session.
  * if you refreshed/cleared unknowingly, you can use the Restore feature to try load previous chat
    session and resume that session. This uses a basic local auto save logic that is in there.

* Using {+}NewChat one can start independent chat sessions.
  * two independent chat sessions are setup by default.

* When you want to print, switching ChatHistoryInCtxt to Full and clicking on the chat session button of
  interest, will display the full chat history till then wrt same, if you want full history for printing.


## Devel note

### Reason behind this

The idea is to be easy enough to use for basic purposes, while also being simple and easily discernable
by developers who may not be from web frontend background (so inturn may not be familiar with template /
end-use-specific-language-extensions driven flows) so that they can use it to explore/experiment things.

And given that the idea is also to help explore/experiment for developers, some flexibility is provided
to change behaviour easily using the devel-tools/console or provided minimal settings ui (wrt few aspects).
Skeletal logic has been implemented to explore some of the end points and ideas/implications around them.

Also by avoiding external packages wrt basic functionality, allows one to have complete control without
having to track external packages in general, while also keeping the size small, especially for embedded
applications, if needed.


### General

Me/gMe->multiChat->simpleChat[chatId].cfg consolidates the settings which control the behaviour into one
object. One can see current settings, as well as change/update them using browsers devel-tool/console.
It is attached to the document object. Some of these can also be updated using the Settings UI.

  * baseURL - the domain-name/ip-address and inturn the port to handshake with the ai engine server.

  * chatProps - maintain a set of properties which manipulate chatting with ai engine

    * apiEP - select between /completions and /chat/completions endpoint provided by the server/ai-model.

    * stream - control between oneshot-at-end and live-stream-as-its-generated collating and showing of the generated response.

      the logic assumes that the text sent from the server follows utf-8 encoding.

      in streaming mode - if there is any exception, the logic traps the same and tries to ensure that text generated till then is not lost.

      * if a very long text is being generated, which leads to no user interaction for sometime and inturn the machine goes into power saving mode or so, the platform may stop network connection, leading to exception.

    * iRecentUserMsgCnt - a simple minded ClientSide SlidingWindow logic to limit context window load at Ai Model end. This is set to 5 by default. So in addition to latest system message, last/latest iRecentUserMsgCnt user messages (after the latest system prompt) and its responses from the ai model along with any associated tool calls will be sent to the ai-model, when querying for a new response. Note that if enabled, only user messages after the latest system message/prompt will be considered.

      This specified sliding window user message count also includes the latest user query.

      * less than 0 : Send entire chat history to server

      * 0 : Send only the system message if any to the server. Even the latest user message wont be sent.

      * greater than 0 : Send the latest chat history from the latest system prompt, limited to specified cnt.

      * NOTE: the latest user message (query/response/...) for which we need a ai response, will also be counted as belonging to the iRecentUserMsgCnt.

    * Markdown

      - enabled: whether auto markdown support is enabled or not at a session level.
        - user can always override explicitly wrt any chat message, as they see fit.
      - always: if true, all messages text content interpreted as Markdown based text and converted to html for viewing.
        if false, then interpret only ai assistant's text content as markdown.
      - htmlSanitize: text content sanitized using browser's dom parser, so html/xml tags get converted to normal visually equivalent text representation, before processing by markdown to html conversion logic.

    * bCompletionFreshChatAlways - whether Completion mode collates complete/sliding-window history when communicating with the server or only sends the latest user query/message.

    * bCompletionInsertStandardRolePrefix - whether Completion mode inserts role related prefix wrt the messages that get inserted into prompt field wrt /Completion endpoint.

    * bTrimGarbage - whether garbage repeatation at the end of the generated ai response, should be trimmed or left as is. If enabled, it will be trimmed so that it wont be sent back as part of subsequent chat history. At the same time the actual trimmed text is shown to the user, once when it was generated, so user can check if any useful info/data was there in the response.

      One may be able to request the ai-model to continue (wrt the last response) (if chat-history is enabled as part of the chat-history-in-context setting), and chances are the ai-model will continue starting from the trimmed part, thus allows long response to be recovered/continued indirectly, in many cases.

      The histogram/freq based trimming logic is currently tuned for english language wrt its is-it-a-alpabetic|numeral-char regex match logic.

  * tools - contains controls related to tool calling

    * enabled - control whether tool calling is enabled or not

      * remember to enable this only for GenAi/LLM models which support tool/function calling.

    * mcpServerUrl - specify the address for the running instance of bundled local.tools/simplemcp.py

    * mcpServerAuth - shared token between simplemcp.py server and client ui, for accessing service provided by it.

      * Shared token is currently hashed with the current year and inturn handshaked over the network. In future if required one could also include a dynamic token provided by simplemcp server during say a special /aum handshake and running counter or so into hashed token. ALERT: However do remember that currently by default handshake occurs over http and not https, so others can snoop the network and get token. Per client ui running counter and random dynamic token can help mitigate things to some extent, if required in future. Remember to enable https mode by specifying a valid public certificate and private key.

    * iResultMaxDataLength - specify what amount of any tool call result should be sent back to the ai engine server.

      * specifying 0 disables this truncating of the results, and inturn full result will be sent to the ai engine server.

    * toolCallResponseTimeoutMS - specifies the time (in msecs) for which the logic should wait for a tool call to respond
    before a default timed out error response is generated and control given back to end user, for them to decide whether
    to submit the error response or wait for actual tool call response further.

    * autoSecs - the amount of time in seconds to wait before the tool call request is auto triggered and generated response is auto submitted back.

      * setting this value to 0 (default), disables auto logic, so that end user can review the tool calls requested by ai and if needed even modify them, before triggering/executing them as well as review and modify results generated by the tool call, before submitting them back to the ai.

      * this is specified in seconds, so that users by default will normally not overload any website through the bundled mcp server.
 
    1. the builtin tools' meta data is sent to the ai model in the requests sent to it.

    2. inturn if the ai model requests a tool call to be made, the same will be done and the response sent back to the ai model, under user control, by default.

    3. as tool calling will involve a bit of back and forth between ai assistant and end user, it is recommended to set iRecentUserMsgCnt to 10 or so, so that enough context is retained during chatting with ai models with tool support. Decide based on your available system and video ram and the type of chat you are having.

  * apiRequestOptions - maintains the list of options/fields to send along with api request, irrespective of whether /chat/completions or /completions endpoint.

    * If you want to add additional options/fields to send to the server/ai-model, and or remove them, for now you can do these actions manually using browser's development-tools/console.

    * For string, numeric, boolean, object fields in apiRequestOptions, including even those added by a user at runtime by directly modifying gMe.apiRequestOptions, setting ui entries will be auto created.

    * cache_prompt option supported by tools/server is allowed to be controlled by user, so that any caching supported wrt system-prompt and chat history, if usable can get used. When chat history sliding window is enabled, cache_prompt logic may or may not kick in at the backend wrt same, based on aspects related to model, positional encoding, attention mechanism etal. However system prompt should ideally get the benefit of caching.

  * headers - maintains the list of http headers sent when request is made to the server. By default

    * Content-Type is set to application/json.

    * Additionally Authorization entry is provided, which can be set if needed using the settings ui.


By using gMe-->simpleChats chatProps.iRecentUserMsgCnt and apiRequestOptions.max_tokens/n_predict one can try
to control the implications of loading of the ai-model's context window by chat history, wrt chat response
to some extent in a simple crude way. You may also want to control the context size enabled when the
server loads ai-model, on the server end. One can look at the current context size set on the server
end by looking at the settings/info block shown when ever one switches-to/is-shown a new session.


Sometimes the browser may be stuborn with caching of the file, so your updates to html/css/js
may not be visible. Also remember that just refreshing/reloading page in browser or for that
matter clearing site data, dont directly override site caching in all cases. Worst case you may
have to change port. Or in dev tools of browser, you may be able to disable caching fully.


The settings are maintained as part of each specific chat session, including the server to communicate with.
So if one changes the server ip/url in setting, then all subsequent chat wrt that session will auto switch
to this new server. And based on the client side sliding window size selected, some amount of your past chat
history from that session will also be sent to this new server.



### Default setup

By default things are setup to try and make the user experience a bit better, if possible.
However a developer when testing the server of ai-model may want to change these value.

Using chatProps.iRecentUserMsgCnt reduce chat history context sent to the server/ai-model to be
just the system-prompt, few prev-user-requests-and-ai-responses and cur-user-request, instead of
full chat history. This way if there is any response with garbage/repeatation, it doesnt
mess with things beyond the next few question/request/query, in some ways. The trim garbage
option also tries to help avoid issues with garbage in the context to an extent.

Set max_tokens to 2048 or as needed, so that a relatively large previous reponse doesnt eat up
the space available wrt next query-response. While parallely allowing a good enough context size
for some amount of the chat history in the current session to influence future answers. However
dont forget that the server when started should also be started with a model context size of
2k or more, as needed.

  The /completions endpoint of tools/server doesnt take max_tokens, instead it takes the
  internal n_predict, for now add the same here on the client side, maybe later add max_tokens
  to /completions endpoint handling code on server side.

NOTE: One may want to experiment with frequency/presence penalty fields in apiRequestOptions
wrt the set of fields sent to server along with the user query, to check how the model behaves
wrt repeatations in general in the generated text response.

A end-user can change these behaviour by editing these through gMe from browser's devel-tool/
console or by using provided settings ui (for settings exposed through ui). The logic uses a
generic helper which autocreates property edit ui elements for specified set of properties. If
new property is a number or text or boolean or a object with properties within it, autocreate
logic will try handle it automatically. A developer can trap this autocreation flow and change
things if needed.


### OpenAi / Equivalent API WebService

One may be abe to handshake with OpenAI/Equivalent api web service's /chat/completions endpoint
for a minimal chatting experimentation by setting the below.

* the baseUrl in settings ui
  * https://api.openai.com/v1 or similar

* Wrt request body - gMe-->simpleChats apiRequestOptions
  * model (settings ui)
  * any additional fields if required in future

* Wrt request headers - gMe-->simpleChats headers
  * Authorization (available through settings ui)
    * Bearer THE_OPENAI_API_KEY
  * any additional optional header entries like "OpenAI-Organization", "OpenAI-Project" or so

NOTE: Not tested, as there is no free tier api testing available. However logically this might
work.


### Tool Calling

Given that browsers provide a implicit env for not only showing ui, but also running logic,
simplechat client ui allows use of tool calling support provided by the newer ai models by
end users of llama.cpp's server in a simple way without needing to worry about seperate mcp
host / router, tools etal, for basic useful tools/functions like calculator, code execution
(javascript in this case), data store, ai calling ai, ...

Additionally if users want to work with web content or pdf content as part of their ai chat
session, Few functions related to web access as well as pdf access which work with a included
python based simple mcp server (rather mcp-ish) have been implemented.

This can allow end users to use some basic yet useful tool calls to enhance their ai chat
sessions to some extent. It also provides for a simple minded exploration of tool calling
support in newer ai models and some fun along the way as well as occasional practical use
like

* verifying mathematical or logical statements/reasoning made by the ai model during chat
sessions by getting it to also create and execute mathematical expressions or code to verify
such stuff and so.

* access content (including html, pdf, text based...) from local file system or the internet
and augment the ai model's context with additional data as needed to help generate better
responses. This can also be used for
  * generating the latest news summary by fetching from news aggregator sites and collating
  organising and summarising the same
  * searching for specific topics and summarising the search results and or fetching and
  analysing found data to generate summary or to explore / answer queries around that data ...
  * or so
  * NOTE: rather here unlike a pure RAG based flow, ai itself helps identify what additional
  data to get and work on and goes about trying to do the same

* save collated data or generated analysis or more to the provided data store and retrieve
them later to augment the analysis / generation then. Also could be used to summarise chat
session till a given point and inturn save the summary into data store and later retrieve
the summary and continue the chat session using the summary and thus with a reduced context
window to worry about.

* use your imagination and ai models capabilities as you see fit, without restrictions from
others.

The tool calling feature has been tested with Gemma3N, Granite4 and GptOss.

ALERT: The simple minded way in which this is implemented, it provides some minimal safety
mechanism like running ai generated code in web workers and restricting web access to user
specified whitelist and so, but it can still be dangerous in the worst case, So remember
to verify all the tool calls requested and the responses generated manually to ensure
everything is fine, during interaction with ai models with tools support. One could also
always run this from a discardable vm, just in case if one wants to be extra cautious.

#### Builtin Tools

The following tools/functions are currently provided by default

##### directly in and using browser capabilities

* sys_date_time - provides the current date and time

* simple_calculator - can solve simple arithmatic expressions

* run_javascript_function_code - can be used to run ai generated or otherwise javascript code
  using browser's js capabilities.

* data_store_get/set/delete/list - allows for a basic data store to be used, to maintain data
  and or context across sessions and so...

* external_ai - allows ai to use an independent fresh by default session of itself / different ai,
  with a custom system prompt of ai's choosing and similarly user message of ai's choosing,
  in order to get any job it deems necessary to be done in a uncluttered indepedent session.
  * in its default configuration, helps ai to process stuff that it needs, without having to worry
    about any previous chat history etal messing with the current data's context and processing.
  * helps ai to process stuff with targeted system prompts of its choosing, for the job at hand.
  * by default
    * tool calling is disabled wrt the external_ai's independent session.
      * it was noticed that else even external_ai may call into more external_ai calls trying to
        find answers to the same question/situation.
      * maybe one can enable tool calling, while explicitly disabling of external_ai tool call
        from within external_ai tool call related session or so later...
    * client side sliding window size is set to 1 so that only system prompt and ai set user message
      gets handshaked with the external_ai instance
    * End user can change this behaviour by changing the corresponding settings of the TCExternalAi
      special chat session, which is internally used for this tool call.
  * Could be used by ai for example to
    * break down the task at hand into sub tasks that need to be carried out
    * summarise a large text content, where it could use the context of the text to generate a
      suitable system prompt for summarising things suitably
    * create a structured data from a raw textual data
    * act as a literary critic or any domain expert as the case may be
    * or so and so and so ...
    * given the fuzzy nature of the generative ai, sometimes the model may even use this tool call
      to get answer to questions like what is your name ;>
  * end user can use this mechanism to try and bring in an instance of ai running on a more powerful
    machine with more compute and memory capabiliteis, but then to be used only if needed or so

Most of the above (except for external ai call) are run from inside web worker contexts. Currently the
ai generated code / expression is run through a simple minded eval inside a web worker mechanism. Use
of WebWorker helps avoid exposing browser global scope to the generated code directly. However any
shared web worker scope isnt isolated.

Either way always remember to cross check tool requests and generated responses when using tool calling.

##### using bundled simplemcp.py (helps bypass browser cors restriction, ...)

* fetch_url_raw - fetch contents of the requested url through/using mcp server

* fetch_html_text - fetch text parts of the html content from the requested url through a mcp server.
  Related logic tries to strip html response of html tags and also head, script, style, header,footer,
  nav, ... blocks (which are usually not needed).

* search_web_text - search for the specified words using the configured search engine and return the
  plain textual content from the search result page.

  From the bundled simplemcp.py one can control the search engine details like

    * template - specify the search engine's search url template along with the tag SEARCHWORDS in place where the search words should be substituted at runtime.

    * drops - allows one to drop contents of html tags with specified id from the final plain text search result.

      * specify a list of dicts, where each dict should contain a 'tag' entry specifying the tag to filter like div or p or ... and also a 'id' entry which specifies the id of interest.

* fetch_pdf_as_text - fetch/read specified pdf file and extract its textual content
  * this depends on the pypdf python based open source library
  * create a outline of titles along with numbering if the pdf contains a outline/toc

* fetch_xml_filtered - fetch/read specified xml file and optionally filter out any specified tags
  * allows one to specify a list of tags related REs,
    to help drop the corresponding tags and their contents fully.
  * to drop a tag, specify regular expression
    * that matches the corresponding heirarchy of tags involved
      * where the tag names should be in lower case and suffixed with :
    * if interested in dropping a tag independent of where it appears use
      * .*:tagname:.*
      * rather the tool call meta data passed to ai model explains the same and provides a sample.

the above set of web related tool calls work by handshaking with a bundled simple local mcp (may be
add caching in future) server logic, this helps bypass the CORS restrictions applied if trying to
directly fetch from the browser js runtime environment.

Local file access is also enabled for web fetch and pdf tool calls, if one uses the file:/// scheme
in the url, so be careful as to where and under which user id the simple mcp will be run.

* one can always disable local file access by removing 'file' from the list of acl.schemes in
simplemcp.json config file.

Implementing some of the tool calls through the simplemcp.py server and not directly in the browser
js env, allows one to isolate the core of these logic within a discardable VM or so and also if required
in a different region or so, by running the simplemcp.py in such a vm.

Depending on path and method specified using json-rpc wrt the mcp server, it executes corresponding logic.

This chat client logic does a simple check to see if bundled simplemcp is running at specified
mcpServerUrl and in turn the provided tool calls like those related to web / pdf etal.

The bundled simple mcp

* can be found at
  * tools/server/public_simplechat/local.tools/simplemcp.py

* it provides for a basic white list of allowed domains to access, to be specified by the end user.
  This should help limit web access to a safe set of sites determined by the end user. There is also
  a provision for shared bearer token to be specified by the end user. One could even control what
  schemes are supported wrt the urls.

* by default runs in http mode. If valid sec.keyfile and sec.certfile options are specified, logic
  will run in https mode.
  * Remember to also update tools->mcpServerUrl wrt the chat session settings.
    * the new url will be used for subsequent tool handshakes, however remember that the list of
      tool calls supported wont get updated, till this chat client web ui is refreshed/reloaded.

* it tries to mimic the client/browser making the request to it by propogating header entries like
  user-agent, accept and accept-language from the got request to the generated request during this
  mcp based proxying, so that websites will hopefully respect the request rather than blindly
  rejecting it as coming from a non-browser entity.

* allows getting specified local or web based pdf files and extract their text content for ai to use

In future it can be further extended to help with other relatively simple yet useful tool calls like
fetch_rss and so.

  * for now fetch_rss can be indirectly achieved using
    * fetch_url_raw or better still
    * xmlfiltered and its tagDropREs

#### Extending with new tools

This client ui implements the json schema based function calling convention supported by gen ai
engines over http.

Provide a descriptive meta data explaining the tool / function being provided for tool calling,
as well as its arguments.

Provide a handler which
* implements the specified tool / function call or
* rather in some cases constructs the code to be run to get the tool / function call job done,
  and inturn pass the same to the provided web worker to get it executed. Use console.log while
  generating any response that should be sent back to the ai model, in your constructed code.
* once the job is done, return the generated result as needed, along with tool call related meta
  data like chatSessionId, toolCallId, toolName which was passed along with the tool call.

Update the tc_switch to include a object entry for the tool, which inturn includes
* the meta data wrt the tool call
* a reference to the handler - handler should take chatSessionId, toolCallId, toolName and toolArgs.
  It should pass these along to the tools web worker, if used.
* the result key (was used previously, may use in future, but for now left as is)

Look into tooljs.mjs, toolai.mjs and tooldb.mjs for javascript and inturn browser web worker based
tool calls and toolweb.mjs for the simplemcp.py based tool calls.

#### OLD: Mapping tool calls and responses to normal assistant - user chat flow

Instead of maintaining tool_call request and resultant response in logically seperate parallel
channel used for requesting tool_calls by the assistant and the resulstant tool role response,
the SimpleChatTC pushes it into the normal assistant - user chat flow itself, by including the
tool call and response as a pair of tagged request with details in the assistant block and inturn
tagged response in the subsequent user block.

This allows GenAi/LLM to be still aware of the tool calls it made as well as the responses it got,
so that it can incorporate the results of the same in the subsequent chat / interactions.

NOTE: This flow tested to be ok enough with Gemma-3N-E4B-it-Q8_0 LLM ai model for now. Logically
given the way current ai models work, most of them should understand things as needed, but need
to test this with other ai models later.

TODO:OLD: Need to think later, whether to continue this simple flow, or atleast use tool role wrt
the tool call responses or even go further and have the logically seperate tool_calls request
structures also.

DONE: rather both tool_calls structure wrt assistant messages and tool role based tool call
result messages are generated as needed now.

#### Related stuff

Promise as well as users of promise (for now fetch) have been trapped wrt their then and catch flow,
so that any scheduled asynchronous code or related async error handling using promise mechanism also
gets executed, before tool calling returns and thus data / error generated by those async code also
get incorporated in result sent to ai engine on the server side.


### Debuging the handshake and beyond

When working with llama.cpp server based GenAi/LLM running locally, to look at the handshake directly
from the commandline, you could run something like below

* sudo tcpdump -i lo -s 0 -vvv -A host 127.0.0.1 and port 8080 | tee /tmp/td.log
* or one could also try look at the network tab in the browser developer console

One could always remove message entries or manipulate chat sessions by accessing document['gMe']
in devel console of the browser

* if you want the last tool call response you submitted to be re-available for tool call execution and
  resubmitting of response fresh, for any reason, follow below steps
  * remove the assistant response from end of chat session, if any, using
    * document['gMe'].multiChat.simpleChats['SessionId'].xchat.pop()
    * [202511] One can even use the del button in the popover menu wrt each chat message to delete
  * reset role of Tool response chat message to TOOL-TEMP from tool
    * toolMessageIndex = document['gMe'].multiChat.simpleChats['SessionId'].xchat.length - 1
    * document['gMe'].multiChat.simpleChats['SessionId'].xchat[toolMessageIndex].role = "TOOL-TEMP"
  * if you dont mind running the tool call again, just deleting the tool response message will also do
  * clicking on the SessionId at top in UI, should refresh the chat ui and inturn it should now give
    the option to control that tool call again
  * this can also help in the case where the chat session fails with context window exceeded
    * you restart the GenAi/LLM server after increasing the context window as needed
    * edit the chat session history as mentioned above, to the extent needed
    * resubmit the last needed user/tool response as needed


## At the end

Also a thank you to all open source and open model developers, who strive for the common good.
