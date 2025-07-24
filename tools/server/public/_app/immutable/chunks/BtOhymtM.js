import"./DsnmJJEf.js";import{a3 as ke,al as Pe,E as Ce,am as Me,an as Ne,M as Se,z as Te,ao as Fe,ap as fe,ab as Ee,O as Le,R as N,p as S,ai as H,a as w,aq as O,b as x,d as T,f as k,c as $,r as b,s as C,t as B,e as V,B as m,aj as L,y as je,ag as me,X as Y,ar as He,as as ze,at as Ae,au as Be,ak as Ie}from"./DHa_N7rM.js";import{a as K,r as D,p as q,b as ne,i as j}from"./Dn_KYLer.js";import{I as W,a as Re,b as qe,s as Z,e as he,d as ge,B as X,f as Oe,g as ae,h as re,j as De,k as ie,l as J,m as de,n as Ke}from"./BrRg135A.js";const We=()=>performance.now(),A={tick:t=>requestAnimationFrame(t),now:()=>We(),tasks:new Set};function pe(){const t=A.now();A.tasks.forEach(e=>{e.c(t)||(A.tasks.delete(e),e.f())}),A.tasks.size!==0&&A.tick(pe)}function Ge(t){let e;return A.tasks.size===0&&A.tick(pe),{promise:new Promise(n=>{A.tasks.add(e={c:t,f:n})}),abort(){A.tasks.delete(e)}}}function le(t,e){fe(()=>{t.dispatchEvent(new CustomEvent(e))})}function Ue(t){if(t==="float")return"cssFloat";if(t==="offset")return"cssOffset";if(t.startsWith("--"))return t;const e=t.split("-");return e.length===1?e[0]:e[0]+e.slice(1).map(n=>n[0].toUpperCase()+n.slice(1)).join("")}function ce(t){const e={},n=t.split(";");for(const a of n){const[o,s]=a.split(":");if(!o||s===void 0)break;const r=Ue(o.trim());e[r]=s.trim()}return e}const Ve=t=>t;function Q(t,e,n,a){var o=(t&Fe)!==0,s="in",r,d=e.inert,p=e.style.overflow,i,l;function h(){return fe(()=>r??=n()(e,a?.()??{},{direction:s}))}var _={is_global:o,in(){e.inert=d,i?.abort(),le(e,"introstart"),i=xe(e,h(),l,1,()=>{le(e,"introend"),i?.abort(),i=r=void 0,e.style.overflow=p})},out(u){{u?.(),r=void 0;return}},stop:()=>{i?.abort()}},c=ke;if((c.transitions??=[]).push(_),Pe){var v=o;if(!v){for(var f=c.parent;f&&(f.f&Ce)!==0;)for(;(f=f.parent)&&(f.f&Me)===0;);v=!f||(f.f&Ne)!==0}v&&Se(()=>{Te(()=>_.in())})}}function xe(t,e,n,a,o){if(Ee(e)){var s,r=!1;return Le(()=>{if(!r){var f=e({direction:"in"});s=xe(t,f,n,a,o)}}),{abort:()=>{r=!0,s?.abort()},deactivate:()=>s.deactivate(),reset:()=>s.reset(),t:()=>s.t()}}if(!e?.duration)return o(),{abort:N,deactivate:N,reset:N,t:()=>a};const{delay:d=0,css:p,tick:i,easing:l=Ve}=e;var h=[];if(i&&i(0,1),p){var _=ce(p(0,1));h.push(_,_)}var c=()=>1-a,v=t.animate(h,{duration:d,fill:"forwards"});return v.onfinish=()=>{v.cancel();var f=1-a,u=a-f,g=e.duration*Math.abs(u),I=[];if(g>0){var F=!1;if(p)for(var M=Math.ceil(g/16.666666666666668),z=0;z<=M;z+=1){var y=f+u*l(z/M),P=ce(p(y,1-y));I.push(P),F||=P.overflow==="hidden"}F&&(t.style.overflow="hidden"),c=()=>{var E=v.currentTime;return f+u*l(E/g)},i&&Ge(()=>{if(v.playState!=="running")return!1;var E=c();return i(E,1-E),!0})}v=t.animate(I,{duration:g,fill:"forwards"}),v.onfinish=()=>{c=()=>a,i?.(a,1-a),o()}},{abort:()=>{v&&(v.cancel(),v.effect=null,v.onfinish=N)},deactivate:()=>{o=N},reset:()=>{},t:()=>c()}}function Xe(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["path",{d:"M12 8V4H8"}],["rect",{width:"16",height:"12",x:"4",y:"8",rx:"2"}],["path",{d:"M2 14h2"}],["path",{d:"M20 14h2"}],["path",{d:"M15 13v2"}],["path",{d:"M9 13v2"}]];W(t,K({name:"bot"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}function Je(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"}],["circle",{cx:"12",cy:"12",r:"3"}]];W(t,K({name:"eye"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}function _e(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["path",{d:"M12 19v3"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2"}],["rect",{x:"9",y:"2",width:"6",height:"13",rx:"3"}]];W(t,K({name:"mic"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}function Qe(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["path",{d:"m16 6-8.414 8.586a2 2 0 0 0 2.829 2.829l8.414-8.586a4 4 0 1 0-5.657-5.657l-8.379 8.551a6 6 0 1 0 8.485 8.485l8.379-8.551"}]];W(t,K({name:"paperclip"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}function Ye(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z"}],["path",{d:"m21.854 2.147-10.94 10.939"}]];W(t,K({name:"send"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}function Ze(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["rect",{width:"20",height:"8",x:"2",y:"2",rx:"2",ry:"2"}],["rect",{width:"20",height:"8",x:"2",y:"14",rx:"2",ry:"2"}],["line",{x1:"6",x2:"6.01",y1:"6",y2:"6"}],["line",{x1:"6",x2:"6.01",y1:"18",y2:"18"}]];W(t,K({name:"server"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}function et(t,e){S(e,!0);/**
 * @license @lucide/svelte v0.525.0 - ISC
 *
 * ISC License
 *
 * Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT). All other copyright (c) for Lucide are held by Lucide Contributors 2022.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */let n=D(e,["$$slots","$$events","$$legacy"]);const a=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2"}]];W(t,K({name:"square"},()=>n,{get iconNode(){return a},children:(o,s)=>{var r=H(),d=w(r);O(d,()=>e.children??N),x(o,r)},$$slots:{default:!0}})),T()}var tt=k("<div><!></div>");function ye(t,e){S(e,!0);let n=q(e,"ref",15,null),a=D(e,["$$slots","$$events","$$legacy","ref","class","children"]);var o=tt();Re(o,r=>({"data-slot":"card",class:r,...a}),[()=>qe("bg-card text-card-foreground flex flex-col gap-6 rounded-xl border py-6 shadow-sm",e.class)]);var s=$(o);O(s,()=>e.children??N),b(o),ne(o,r=>n(r),()=>n()),x(t,o),T()}var at=k('<div class="text-xs opacity-70"> </div>'),rt=k('<div class="whitespace-pre-wrap text-sm"> </div> <!>',1),st=k("<div><!></div>");function nt(t,e){S(e,!0);let n=q(e,"class",3,"");var a=st(),o=$(a);{let s=L(()=>e.message.role==="user"?"bg-primary text-primary-foreground":"bg-muted");ye(o,{get class(){return`max-w-[80%] gap-2 px-4 py-3 ${m(s)??""}`},children:(r,d)=>{var p=rt(),i=w(p),l=$(i,!0);b(i);var h=C(i,2);{var _=c=>{var v=at(),f=$(v,!0);b(v),B(u=>V(f,u),[()=>new Date(e.message.timestamp).toLocaleTimeString()]),x(c,v)};j(h,c=>{e.message.timestamp&&c(_)})}B(()=>V(l,e.message.content)),x(r,p)},$$slots:{default:!0}})}b(a),B(()=>Z(a,1,`flex gap-3 ${n()??""} ${e.message.role==="user"?"justify-end":"justify-start"}`)),x(t,a),T()}var ot=k('<div class="flex items-center space-x-2"><div class="flex space-x-1"><div class="h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.3s]"></div> <div class="h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.15s]"></div> <div class="h-2 w-2 animate-bounce rounded-full bg-current"></div></div> <span class="text-muted-foreground text-sm">Thinking...</span></div>'),it=k('<div><div class="bg-background flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow"><!></div> <!></div>');function dt(t,e){var n=it(),a=$(n),o=$(a);Xe(o,{class:"h-4 w-4"}),b(a);var s=C(a,2);ye(s,{class:"bg-muted max-w-[80%] p-3",children:(r,d)=>{var p=ot();x(r,p)},$$slots:{default:!0}}),b(n),B(()=>Z(n,1,`flex justify-start gap-3 ${e.class??""}`)),x(t,n)}var lt=k('<div><div class="bg-background flex-1 overflow-y-auto p-4"><div class="mb-48 mt-16 space-y-4"><!> <!></div></div></div>');function ct(t,e){S(e,!0);let n=q(e,"messages",19,()=>[]),a=q(e,"isLoading",3,!1),o=me(void 0);je(()=>{m(o)&&(n().length>0||a())&&setTimeout(()=>{m(o)&&(m(o).scrollTop=m(o).scrollHeight)},0)});var s=lt(),r=$(s),d=$(r),p=$(d);he(p,17,n,ge,(h,_)=>{nt(h,{class:"mx-auto w-full max-w-[56rem]",get message(){return m(_)}})});var i=C(p,2);{var l=h=>{dt(h,{class:"mx-auto w-full max-w-[56rem]"})};j(i,h=>{a()&&h(l)})}b(d),b(r),ne(r,h=>Y(o,h),()=>m(o)),b(s),B(()=>Z(s,1,`flex h-full flex-col ${e.class??""}`)),x(t,s),T()}function vt(t){t&&(t.style.height="auto",t.style.height=t.scrollHeight+"px")}var ut=t=>vt(t.currentTarget),ft=k("<!> <!>",1),mt=k(`<div class="mt-2 flex items-center justify-center"><p class="text-muted-foreground text-xs">Press <kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Enter</kbd> to
				send, <kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for new
				line</p></div>`),ht=k('<form><div class="bg-muted/30 border-border/40 focus-within:border-primary/40 flex-column relative min-h-[48px] items-center rounded-3xl border px-5 py-3 shadow-sm transition-all focus-within:shadow-md"><div class="flex-1"><textarea placeholder="Ask anything..." class="placeholder:text-muted-foreground text-md max-h-32 min-h-[24px] w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none focus-visible:ring-0 focus-visible:ring-offset-0"></textarea></div> <div class="flex items-center justify-between gap-1"><!> <div><!></div></div></div> <!></form>');function ve(t,e){S(e,!0);let n=q(e,"disabled",3,!1),a=q(e,"isLoading",3,!1),o=q(e,"showHelperText",3,!0),s=me(""),r;function d(y){y.preventDefault(),!(!m(s).trim()||n()||a())&&(e.onsend?.(m(s).trim()),Y(s,""),r&&(r.style.height="auto"))}function p(y){if(y.key==="Enter"&&!y.shiftKey){if(y.preventDefault(),!m(s).trim()||n()||a())return;e.onsend?.(m(s).trim()),Y(s,""),r&&(r.style.height="auto")}}function i(){e.onstop?.()}var l=ht(),h=$(l),_=$(h),c=$(_);ze(c),c.__keydown=p,c.__input=[ut],ne(c,y=>r=y,()=>r),b(_);var v=C(_,2),f=$(v);{let y=L(()=>n()||a());X(f,{type:"button",variant:"ghost",class:"text-muted-foreground hover:text-foreground h-9 w-9 rounded-full p-0",get disabled(){return m(y)},children:(P,E)=>{Qe(P,{class:"h-4 w-4"})},$$slots:{default:!0}})}var u=C(f,2),g=$(u);{var I=y=>{X(y,{type:"button",variant:"ghost",onclick:i,class:"text-muted-foreground hover:text-destructive h-9 w-9 rounded-full p-0",children:(P,E)=>{et(P,{class:"h-6 w-6"})},$$slots:{default:!0}})},F=y=>{var P=ft(),E=w(P);{let G=L(()=>n()||a());X(E,{type:"button",variant:"ghost",class:"text-muted-foreground hover:text-foreground h-9 w-9 rounded-full p-0",get disabled(){return m(G)},children:(U,R)=>{_e(U,{class:"h-6 w-6"})},$$slots:{default:!0}})}var ee=C(E,2);{let G=L(()=>!m(s).trim()||n()||a());X(ee,{type:"submit",get disabled(){return m(G)},class:"h-9 w-9 rounded-full p-0",children:(U,R)=>{Ye(U,{class:"h-6 w-6"})},$$slots:{default:!0}})}x(y,P)};j(g,y=>{a()?y(I):y(F,!1)})}b(u),b(v),b(h);var M=C(h,2);{var z=y=>{var P=mt();x(y,P)};j(M,y=>{o()&&y(z)})}b(l),B(()=>{Z(l,1,`bg-background border-radius-bottom-none mx-auto max-w-4xl overflow-hidden rounded-3xl ${e.class??""}`),c.disabled=n()}),Ae("submit",l,d),Oe(c,()=>m(s),y=>Y(s,y)),x(t,l),T()}He(["keydown","input"]);var gt=k("<!> ",1),pt=k("<!> ",1),xt=k('<div class="flex items-center justify-center gap-3 text-sm text-muted-foreground"><!> <!> <!></div>');function _t(t,e){S(e,!0);const n=L(()=>ae.serverProps),a=L(()=>ae.modelName),o=L(()=>ae.supportedModalities);var s=H(),r=w(s);{var d=p=>{var i=xt(),l=$(i);{var h=u=>{re(u,{variant:"outline",class:"text-xs",children:(g,I)=>{var F=gt(),M=w(F);Ze(M,{class:"mr-1 h-3 w-3"});var z=C(M);B(()=>V(z,` ${m(a)??""}`)),x(g,F)},$$slots:{default:!0}})};j(l,u=>{m(a)&&u(h)})}var _=C(l,2);{var c=u=>{re(u,{variant:"secondary",class:"text-xs",children:(g,I)=>{Be();var F=Ie();B(M=>V(F,`ctx: ${M??""}`),[()=>m(n).n_ctx.toLocaleString()]),x(g,F)},$$slots:{default:!0}})};j(_,u=>{m(n).n_ctx&&u(c)})}var v=C(_,2);{var f=u=>{var g=H(),I=w(g);he(I,17,()=>m(o),ge,(F,M)=>{re(F,{variant:"secondary",class:"text-xs",children:(z,y)=>{var P=pt(),E=w(P);{var ee=R=>{Je(R,{class:"mr-1 h-3 w-3"})},G=R=>{var oe=H(),be=w(oe);{var we=te=>{_e(te,{class:"mr-1 h-3 w-3"})};j(be,te=>{m(M)==="audio"&&te(we)},!0)}x(R,oe)};j(E,R=>{m(M)==="vision"?R(ee):R(G,!1)})}var U=C(E);B(()=>V(U,` ${m(M)??""}`)),x(z,P)},$$slots:{default:!0}})}),x(u,g)};j(v,u=>{m(o).length>0&&u(f)})}b(i),x(p,i)};j(r,p=>{m(n)&&p(d)})}x(t,s),T()}function $e(t){const e=t-1;return e*e*e+1}function ue(t){const e=typeof t=="string"&&t.match(/^\s*(-?[\d.]+)([^\s]*)\s*$/);return e?[parseFloat(e[1]),e[2]||"px"]:[t,"px"]}function yt(t,{delay:e=0,duration:n=400,easing:a=$e,x:o=0,y:s=0,opacity:r=0}={}){const d=getComputedStyle(t),p=+d.opacity,i=d.transform==="none"?"":d.transform,l=p*(1-r),[h,_]=ue(o),[c,v]=ue(s);return{delay:e,duration:n,easing:a,css:(f,u)=>`
			transform: ${i} translate(${(1-f)*h}${_}, ${(1-f)*c}${v});
			opacity: ${p-l*u}`}}function se(t,{delay:e=0,duration:n=400,easing:a=$e,axis:o="y"}={}){const s=getComputedStyle(t),r=+s.opacity,d=o==="y"?"height":"width",p=parseFloat(s[d]),i=o==="y"?["top","bottom"]:["left","right"],l=i.map(g=>`${g[0].toUpperCase()}${g.slice(1)}`),h=parseFloat(s[`padding${l[0]}`]),_=parseFloat(s[`padding${l[1]}`]),c=parseFloat(s[`margin${l[0]}`]),v=parseFloat(s[`margin${l[1]}`]),f=parseFloat(s[`border${l[0]}Width`]),u=parseFloat(s[`border${l[1]}Width`]);return{delay:e,duration:n,easing:a,css:g=>`overflow: hidden;opacity: ${Math.min(g*20,1)*r};${d}: ${g*p}px;padding-${i[0]}: ${g*h}px;padding-${i[1]}: ${g*_}px;margin-${i[0]}: ${g*c}px;margin-${i[1]}: ${g*v}px;border-${i[0]}-width: ${g*f}px;border-${i[1]}-width: ${g*u}px;min-${d}: 0`}}var $t=k('<div class="flex h-full flex-col"><div class="flex-1 overflow-hidden"><!> <div class="sticky bottom-0 m-auto max-w-[56rem]"><div class="bg-background m-auto rounded-t-3xl border-t pb-4"><!></div></div></div></div>'),bt=k('<div class="flex h-full items-center justify-center"><div class="w-full max-w-2xl px-4"><div class="mb-8 text-center"><h1 class="mb-2 text-3xl font-semibold tracking-tight">llama.cpp</h1> <p class="text-muted-foreground text-lg">How can I help you today?</p></div> <div class="mb-6 flex justify-center"><!></div> <div><!></div></div></div>');function Mt(t,e){S(e,!0);let n=q(e,"showCenteredEmpty",3,!1);const a=L(()=>n()&&!De()&&ie().length===0&&!J());async function o(i){await Ke(i)}var s=H(),r=w(s);{var d=i=>{var l=$t(),h=$(l),_=$(h);{let u=L(ie),g=L(J);ct(_,{class:"mb-36",get messages(){return m(u)},get isLoading(){return m(g)}})}var c=C(_,2),v=$(c),f=$(v);{let u=L(J);ve(f,{get isLoading(){return m(u)},showHelperText:!1,onsend:o,onstop:()=>de()})}b(v),b(c),b(h),b(l),Q(1,c,()=>se,()=>({duration:400,axis:"y"})),x(i,l)},p=i=>{var l=bt(),h=$(l),_=$(h),c=C(_,2),v=$(c);_t(v,{}),b(c);var f=C(c,2),u=$(f);{let g=L(J);ve(u,{get isLoading(){return m(g)},showHelperText:!0,onsend:o,onstop:()=>de()})}b(f),b(h),b(l),Q(1,_,()=>yt,()=>({y:-30,duration:600})),Q(1,c,()=>se,()=>({duration:500,delay:300,axis:"y"})),Q(1,f,()=>se,()=>({duration:600,delay:500,axis:"y"})),x(i,l)};j(r,i=>{m(a)?i(p,!1):i(d)})}x(t,s),T()}export{Mt as C};
