"use strict";
const D=JSON.parse(document.getElementById("report-data").textContent);
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const names={student_before:"正式 SFT 起点",student_after:"OPSD · 32 步",student_after_grpo_hf:"GRPO · HF 生成训练",student_after_grpo_vllm:"GRPO · vLLM 生成训练",teacher_before:"特权教师（预检）"};
const studentStages=Object.keys(names).slice(0,4);
const fmt=(v,n=3)=>Number(v).toFixed(n);
const seg=s=>s?`[${fmt(s[0])}, ${fmt(s[1])}] s`:"无区间";
let selectedCase=0,selectedStage="student_before",selectedChart="opsd_hf";

function source(id, key){const s=D.sources[key];$(id).innerHTML=`<div class="source-label">${esc(s.file)} · 第 ${s.line} 行起</div><pre class="code-source">${esc(s.text)}</pre>`;}
function intervals(c){const t=c.target,duration=t.proposal[1]-t.proposal[0];const list=[{name:"GT · 片段内真值",segment:t.target_relative,key:"gt"},...studentStages.map(k=>({name:names[k],segment:c.predictions[k].relative_segment,key:k}))];$("intervals").innerHTML=list.map(r=>{const pred=r.key==="gt"?null:c.predictions[r.key];return `<div class="interval-row ${r.key==="gt"?"gt":""} ${r.key===selectedStage?"selected":""}"><div class="interval-title"><span>${r.name}</span><span>${r.segment?seg(r.segment):esc(pred?pred.status:t.target_kind)}</span></div><div class="track">${r.segment?`<div class="segment" style="left:${100*r.segment[0]/duration}%;width:${100*(r.segment[1]-r.segment[0])/duration}%"></div>`:""}</div></div>`;}).join("")+`<div class="axis"><span>0 s</span><span>${fmt(duration,1)} s</span></div>`;}
function renderStage(){const c=D.cases[selectedCase],r=c.predictions[selectedStage];$("stage-tabs").innerHTML=studentStages.map(k=>`<button data-stage="${k}" aria-pressed="${k===selectedStage}">${names[k]}</button>`).join("");$("answer-title").textContent=names[selectedStage]+" · 训练后重新推理";$("answer-metric").textContent=r.iou===null?`预测 ${r.status} / GT real`:`${r.status} · IoU ${fmt(r.iou)}`;$("answer").textContent=r.raw;intervals(c);}
function renderCase(){const c=D.cases[selectedCase],t=c.target,duration=t.proposal[1]-t.proposal[0];const titles=["主例：定位区间由过短变为接近 GT","补例：GRPO 改善，但并非每阶段都进步","反例：real 片段仍被当作异常"];
document.querySelectorAll(".sample-select").forEach(el=>{el.value=String(selectedCase);});
$("original-video").pause();
$("case-tabs").innerHTML=["主例 · OPSD 改善","补例 · GRPO 改善","反例 · real 误判"].map((v,i)=>`<button data-case="${i}" aria-pressed="${i===selectedCase}">${v}</button>`).join("");
$("case-title").textContent=titles[selectedCase];const v=studentStages.slice(0,3).map(k=>c.predictions[k].iou);$("case-note").textContent=t.target_kind==="fake"?`SFT → OPSD → GRPO HF 的 IoU：${v.map(x=>fmt(x)).join(" → ")}。${selectedCase===0?"OPSD 后明显改善，GRPO 保持该区间。":"OPSD 略退步，GRPO HF 后改善；vLLM 分支也保留原始结果供比较。"}`:"标注为 real，模型仍输出了异常解释或区间。教师知道该片段正常，但学生短程训练后没有学会正确拒绝误报。";
$("case-id").textContent=c.id;$("clip").src=c.video;$("original").hidden=selectedCase!==0;if(selectedCase===0){$("original-video").src=c.original_video;$("original-note").textContent=`所选 proposal 位于原视频 ${seg(t.proposal)}。可用播放器查看完整上下文。`;}
const absolute=t.target_relative?t.target_relative.map(x=>x+t.proposal[0]):null;$("sample-meta").innerHTML=[["原视频上的 proposal",seg(t.proposal)],["proposal 时长",fmt(duration,1)+" s"],["GT 标签",t.target_kind],["交集映射回原视频",seg(absolute)]].map(([label,value])=>`<div class="fact"><small>${label}</small><b>${esc(value)}</b></div>`).join("");
renderStage();
const W=1000;$("sampling").innerHTML=`<svg viewBox="0 0 1000 90" role="img" aria-label="40帧实际非均匀采样时间"><rect x="0" y="12" width="200" height="44" fill="#F7F7F8"/><rect x="800" y="12" width="200" height="44" fill="#F7F7F8"/><line x1="0" x2="1000" y1="56" y2="56" stroke="#d9dbe1"/>${c.timestamps.map((ms,i)=>`<line x1="${ms/(duration*1000)*W}" x2="${ms/(duration*1000)*W}" y1="${i<16||i>=24?16:30}" y2="56" stroke="#002FA7" stroke-width="2"><title>第 ${i+1} 帧：${fmt(ms/1000)} s</title></line>`).join("")}<text x="0" y="80" font-size="12">0 s</text><text x="200" y="80" font-size="12">${fmt(duration*.2,2)} s</text><text x="800" y="80" font-size="12">${fmt(duration*.8,2)} s</text><text x="1000" text-anchor="end" y="80" font-size="12">${fmt(duration,2)} s</text></svg>`;
$("timing-example").textContent="这条样本的真实毫秒列表：\n["+c.timestamps.join(", ")+"]\n\nframes_indices = 以上列表\nfps = 1000.0\ndo_sample_frames = False";
$("template-preview").textContent=c.encoded_prompts.student.text;
$("patches").innerHTML=c.encoded_prompts.student.times.map((time,i)=>`<div class="${i<8||i>=12?"boundary":""}" title="帧 ${2*i+1} / ${2*i+2}：${c.timestamps[2*i]} / ${c.timestamps[2*i+1]} ms"><small>帧 ${2*i+1}–${2*i+2}</small><b>${esc(time)} s</b></div>`).join("");
$("patch-example").textContent=`例如第 9 组：帧 17、18 的时间为 ${c.timestamps[16]}、${c.timestamps[17]} ms，平均 ${fmt((c.timestamps[16]+c.timestamps[17])/2000,3)} s；模板实际显示 <${c.encoded_prompts.student.times[8]} seconds>。这一步做时间分组与显示取整，不是抹掉非均匀采样。`;
$("student-prompt").textContent=c.opsd.messages[0].content;$("teacher-prompt").textContent=c.opsd.teacher_prompt;$("student-check").textContent=c.predictions.student_before.raw;$("teacher-check").textContent=c.predictions.teacher_before.raw;
$("teacher-heading").textContent=t.target_kind==="fake"?"fake：真值区间 + 对象与异常线索":"real：明确正常，描述普通事件";
$("teacher-info").textContent=t.target_kind==="fake"?`该片段为 fake，训练目标区间 ${seg(t.target_relative)}。教师输入包含真假、区间与解释线索，学生不看这些字段。`:"该片段为 real。教师被告知正常，并要求不要把普通事件解释成异常，最后输出 Real。";
$("student-encoded").textContent=`${c.encoded_prompts.student.tokens} 个输入 token\n\n`+c.encoded_prompts.student.text;$("teacher-encoded").textContent=`${c.encoded_prompts.teacher.tokens} 个输入 token\n\n`+c.encoded_prompts.teacher.text;
const g=c.group;$("group-note").textContent=`下面是该 proposal 在 GRPO HF 第 ${g.step} 步的一组真实 rollout（选用它已记录组中总奖励标准差最大的一组）。这是训练中采样的四条回答，不是上方 checkpoint 推理的四个阶段。`;$("rollouts").innerHTML=g.answers.map((a,i)=>`<article class="rollout"><h3>回答 ${i+1}<span class="badge ${a.advantage>0?"blue":""}">优势 ${a.advantage>=0?"+":""}${fmt(a.advantage)}</span></h3><pre>${esc(a.raw)}</pre><div class="reward-numbers">${[["定位",a.loc],["格式",a.format],["解释",a.explanation],["加权总分",a.reward]].map(([k,v])=>`<div><small>${k}</small><b>${fmt(v)}</b></div>`).join("")}</div></article>`).join("");
$("advantage-example").textContent=`本组总奖励均值 ${fmt(g.mean,4)}，样本标准差 ${fmt(g.std,4)}。四个优势：${g.answers.map(a=>fmt(a.advantage)).join("、")}；页面重算已与训练日志核对。`;
$("reward-truth").textContent=JSON.stringify({target_kind:c.grpo.target_kind,target_relative:c.grpo.target_relative,clip_duration:c.grpo.clip_duration,evidence:c.grpo.evidence},null,2);
}
function plot(points,label){if(points.length<2)return `<p>此指标没有足够的已记录数据。</p>`;const w=500,h=205,l=52,r=16,t=16,b=35;const ys=points.map(x=>x[1]),xs=points.map(x=>x[0]);let lo=Math.min(...ys),hi=Math.max(...ys);if(hi===lo){hi+=.01;lo-=.01;}const pad=(hi-lo)*.12;lo-=pad;hi+=pad;const minx=Math.min(...xs),maxx=Math.max(...xs);const X=x=>l+(x-minx)/(maxx-minx)*(w-l-r),Y=y=>h-b-(y-lo)/(hi-lo)*(h-t-b);let grid="";for(let i=0;i<4;i++){const y=lo+(hi-lo)*i/3;grid+=`<line x1="${l}" x2="${w-r}" y1="${Y(y)}" y2="${Y(y)}" stroke="#d9dbe1"/><text x="${l-8}" y="${Y(y)+4}" text-anchor="end">${fmt(y,Math.abs(y)<.1?4:2)}</text>`;}return `<svg class="plot" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}，逐步原始数据，无平滑">${grid}<polyline fill="none" stroke="#002FA7" stroke-width="2.2" points="${points.map(p=>`${X(p[0])},${Y(p[1])}`).join(" ")}"/>${points.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.5" fill="#002FA7"><title>step ${p[0]}: ${p[1]}</title></circle>`).join("")}<text x="${l}" y="${h-8}">step ${minx}</text><text x="${w-r}" y="${h-8}" text-anchor="end">${maxx}</text></svg>`;}
function chart(){const labels={sft:"SFT · 独立短跑",opsd_hf:"OPSD · HF",grpo_hf:"GRPO · HF",grpo_vllm:"GRPO · vLLM"};$("chart-tabs").innerHTML=Object.keys(labels).map(k=>`<button data-chart="${k}" aria-pressed="${k===selectedChart}">${labels[k]}</button>`).join("");const r=D.training[selectedChart],p=r.config;const xy=key=>r.logs.filter(r=>Number.isFinite(r[key])&&r["global_step/max_steps"]).map(r=>[parseInt(r["global_step/max_steps"].split("/")[0]),r[key]]);$("chart-summary").textContent=`${labels[selectedChart]} · ${p.steps} 步 · global batch ${p.global_batch_size} · 学习率 ${p.learning_rate} · LoRA r=${p.lora_rank}, α=${p.lora_alpha}。曲线为逐步原始记录，无平滑。`;
$("loss-title").textContent=selectedChart==="sft"?"标准回答交叉熵":selectedChart==="opsd_hf"?"完整词表 JSD":"GRPO 策略目标（可为负）";$("loss-chart").innerHTML=plot(xy("loss"),$("loss-title").textContent);const second=selectedChart.startsWith("grpo")?"reward":"grad_norm";$("secondary-title").textContent=second==="reward"?"总奖励均值":"梯度范数";$("secondary-chart").innerHTML=plot(xy(second),$("secondary-title").textContent);$("original-chart").src=r.chart;}
$("case-tabs").addEventListener("click",e=>{const b=e.target.closest("button[data-case]");if(b){selectedCase=Number(b.dataset.case);renderCase();}});$("stage-tabs").addEventListener("click",e=>{const b=e.target.closest("button[data-stage]");if(b){selectedStage=b.dataset.stage;renderStage();}});$("chart-tabs").addEventListener("click",e=>{const b=e.target.closest("button[data-chart]");if(b){selectedChart=b.dataset.chart;chart();}});
$("print").addEventListener("click",()=>window.print());
document.querySelectorAll(".sample-select").forEach(el=>el.addEventListener("change",()=>{selectedCase=Number(el.value);renderCase();}));
$("selection").textContent=D.selection;
$("download").href=location.href.split("#")[0];
$("sft-prompt").textContent=D.cases[0].sft.messages[0].content;$("sft-answer").textContent=D.cases[0].sft.messages[1].content;
$("token-summary").textContent=`主例经实际训练模板编码：${D.tokens.total} 个 token；${D.tokens.supervised} 个有效回答 label；${D.tokens.ignored} 个位置忽略。首次有效 label 位于索引 ${D.tokens.first}（从 0 开始）。`;
$("token-table").innerHTML=D.tokens.rows.map(r=>`<tr class="${r.label!==-100?"highlight":""}"><td>${r.position}</td><td>${r.id}</td><td><code>${esc(JSON.stringify(r.text))}</code></td><td>${r.label}</td><td>${r.label!==-100?"计算 loss":"忽略"}</td></tr>`).join("");
$("metrics").innerHTML=D.metrics.map(r=>`<tr class="${r.stage==="student_after"?"highlight":""}"><td>${names[r.stage]}</td><td>${fmt(r.iou,4)}</td><td>${r.real_correct} / ${r.real_count}</td><td>${r.invalid} / ${D.sample_count}</td></tr>`).join("");
function bars(id,key,label,unit){const rs=D.speed.runs,max=Math.max(...rs.map(r=>r[key]));$(id).innerHTML=`<h4 class="bar-heading">${label}</h4>`+rs.map((r,i)=>`<div class="barline"><p><span>${i===0?"Transformers":"vLLM"}</span><b>${fmt(r[key],2)} ${unit}</b></p><div class="track"><div class="segment" style="width:${r[key]/max*100}%"></div></div></div>`).join("");}bars("speed-bars","mean_step_seconds","平均每步耗时 · 越低越快","s");bars("memory-bars","peak_observed_device_used_gib","设备显存采样峰值","GiB");
$("hero-speed").textContent=fmt(D.speed.runs[0].mean_step_seconds/D.speed.runs[1].mean_step_seconds,2)+"×";
[["proposal-source","proposal"],["template-source","template"],["jsd-source","jsd"],["grpo-source","grpo"],["kl-source","kl"],["reward-source","reward"],["inference-source","inference"],["ray-sync-source","ray_sync"]].forEach(([id,k])=>source(id,k));
$("provenance").textContent=D.provenance.map(r=>r.path+"\nSHA256 "+r.sha256).join("\n\n");
$("reward-caveat").textContent="从 real 反例能看到一个奖励设计问题：越界回答定位 −0.5、格式 −0.1，总分 −0.6；有效误报区间定位 −1、格式 +0.1，总分 −0.9。前者虽然也错，却能拿到正优势。当前页面忠实展示这一行为，没有修改训练奖励；正式实验应检查这种相对排序是否符合目标。";
// Keep all saved checkpoint outputs available, beyond the three media examples.
$("batch-counts").innerHTML=`<tr><td>SFT</td><td>通常 8 条；每轮末批 6 条</td><td>${D.training.sft.data_visits} 次样本使用（实际 epoch 进度）</td></tr><tr><td>OPSD</td><td>8 条样本</td><td>${D.training.opsd_hf.data_visits} 次样本使用，重复 4 轮</td></tr><tr><td>GRPO 的每个分支</td><td>2 个 proposal 组 × 每组 4 条回答</td><td>${D.training.grpo_hf.data_visits} 个组、${D.training.grpo_hf.data_visits*D.training.grpo_hf.config.num_generations} 条回答</td></tr>`;
const branches=studentStages.slice(2);
$("trajectory-summary").innerHTML=`<div class="table-scroll"><table><thead><tr><th>GRPO 训练分支</th><th>SFT → OPSD → GRPO<br>两次均提高 IoU</th><th>GRPO 比 OPSD<br>提高 IoU</th><th>SFT 误报 fake<br>最终正确输出 Real</th></tr></thead><tbody>${branches.map(k=>{const r=D.trajectory[k];return `<tr><td>${names[k]}</td><td><b>${r.continuous.length}</b> / ${r.fake_count}</td><td><b>${r.grpo_improved.length}</b> / ${r.fake_count}</td><td><b>${r.real_corrected.length}</b> / ${r.real_count}</td></tr>`;}).join("")}</tbody></table></div><p>OPSD 后正确输出 Real：${D.inference.filter(r=>r.kind==="real"&&r.predictions.student_before.status==="fake"&&r.predictions.student_after.status==="real").length} 条。下方保留“OPSD 改善并保持”“GRPO 后改善”“real 仍误判”三种真实例子。</p>`;
let auditSelection=null;
function auditAnswers(index){
  auditSelection=index;
  const r=D.inference[index];
  $("audit-answer").innerHTML=`<h4>${esc(r.id)}</h4><p>GT：${esc(r.kind)}${r.target_relative?" · "+seg(r.target_relative):""}；原视频 proposal：${seg(r.proposal)}。</p><div class="two">${studentStages.map(k=>{const p=r.predictions[k];return `<article class="panel"><span class="evidence-label">训练后推理</span><h4>${names[k]}</h4><p>${esc(p.status)}${p.iou===null?"":" · IoU "+fmt(p.iou,4)}${p.relative_segment?" · "+seg(p.relative_segment):""}</p><pre>${esc(p.raw)}</pre></article>`;}).join("")}</div>`;
  $("audit-rows").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",Number(b.dataset.audit)===index));
}
function auditTable(){
  const filter=$("audit-filter").value;
  const chosen=D.inference.map((r,index)=>({r,index})).filter(({r})=>{
    if(filter==="all") return true;
    if(filter==="fake"||filter==="real") return r.kind===filter;
    if(filter==="continuous") return branches.some(k=>D.trajectory[k].continuous.includes(r.id));
    const branch=filter==="hf-gain"?branches[0]:branches[1];
    return D.trajectory[branch].grpo_improved.includes(r.id);
  });
  $("audit-count").textContent=`${chosen.length} / ${D.sample_count} 条`;
  $("audit-rows").innerHTML=chosen.length?chosen.map(({r,index})=>`<tr><td><button data-audit="${index}" title="${esc(r.id)}" aria-pressed="false">${index+1}. ${esc(r.id.split("/").pop())}</button></td><td>${r.kind}</td>${studentStages.map(k=>{const p=r.predictions[k];return `<td>${r.kind==="real"?esc(p.status):fmt(p.iou,4)+(p.status!=="fake"?`<small>${esc(p.status)}</small>`:"")}</td>`;}).join("")}</tr>`).join(""):'<tr><td colspan="6">这 64 条记录中，没有符合此条件的样本。</td></tr>';
  if(chosen.length) auditAnswers(chosen.some(x=>x.index===auditSelection)?auditSelection:chosen[0].index);
  else {auditSelection=null;$("audit-answer").innerHTML="";}
}
$("audit-filter").addEventListener("change",auditTable);
$("audit-rows").addEventListener("click",e=>{const b=e.target.closest("button[data-audit]");if(b)auditAnswers(Number(b.dataset.audit));});

// This diagram represents the implemented barriers, not measured stage timings.
const roleDescriptions={
  opsd:[
    ["Rollout · 可用 HF / vLLM","学生生成", "视频 + 普通提示 + 学生 v", "学生回答 token，附样本 ID 与版本 v"],
    ["Scorer · 固定教师","教师前向", "同一视频 + 特权提示 + 同一学生回答", "每个回答位置的完整词表 logits；不更新教师"],
    ["Trainer · HF + DDP","学生前向与更新", "普通输入 + 学生回答 + 教师 logits", "完整词表 JSD → 反传 → 学生 v+1"]
  ],
  grpo:[
    ["Rollout · 可用 HF / vLLM","每个 proposal 生成 4 条", "视频 + 普通提示 + 学生 v", "4 条回答 token，附样本 ID 与版本 v"],
    ["Scorer + CPU Reward","参考前向与奖励", "同一回答；奖励端另读取 GT 与标注证据", "固定参考 token log-prob；3 项奖励 → 组内优势"],
    ["Trainer · HF + DDP","策略更新", "回答 + 参考概率 + 优势；更新前前向并固定旧概率", "带梯度前向 → clipped 策略目标 + KL → 学生 v+1"]
  ]
};
function roleFlow(stage){
  $("role-tabs").querySelectorAll("button").forEach(b=>b.setAttribute("aria-pressed",b.dataset.roleStage===stage));
  $("role-flow").innerHTML=roleDescriptions[stage].map(([role,title,input,output],i)=>`<article><span class="role-number">${i+1}</span><small>${role}</small><h4>${title}</h4><dl><dt>接收</dt><dd>${input}</dd><dt>交出</dt><dd>${output}</dd></dl></article>`).join("");
}
$("role-tabs").addEventListener("click",e=>{const b=e.target.closest("button[data-role-stage]");if(b)roleFlow(b.dataset.roleStage);});
$("eval-config").textContent=branches.map(k=>{const r=D.inference_runs[k];return `${names[k]}\nmode: ${r.mode}\nmodel: ${r.weights.model}\nadapter: ${r.weights.adapter.path}\nmax_new_tokens: ${r.max_new_tokens}\ninspect_model.py SHA256: ${r.source["Qwen/demo/inspect_model.py"]}`;}).join("\n\n");
const [hfRun,vllmRun]=D.speed.runs;
$("speed-conclusion").textContent=`平均一步 ${fmt(hfRun.mean_step_seconds,2)} s → ${fmt(vllmRun.mean_step_seconds,2)} s，约 ${fmt(hfRun.mean_step_seconds/vllmRun.mean_step_seconds,2)}× 速度，耗时减少 ${fmt((1-vllmRun.mean_step_seconds/hfRun.mean_step_seconds)*100,1)}%。设备显存采样峰值增加 ${fmt(vllmRun.peak_observed_device_used_gib-hfRun.peak_observed_device_used_gib,2)} GiB。`;
auditTable();roleFlow("opsd");renderCase();chart();
const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){document.querySelectorAll("nav a").forEach(a=>a.classList.toggle("current",a.hash==="#"+e.target.id));}},{rootMargin:"-15% 0px -65% 0px"});document.querySelectorAll("main>section[id]").forEach(s=>observer.observe(s));
