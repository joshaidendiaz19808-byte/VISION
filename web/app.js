// VISION app: one screen, talks to the VISION API. Everything shown is built with textContent,
// so nothing from a file, an account or a repo can inject markup.
"use strict";

const API = (window.VISION_API || "").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
const COLORS = { "LOOKS REAL": "var(--real)", "QUESTIONABLE": "var(--warn)", "LOOKS FAKE": "var(--fake)",
                 "NOT ENOUGH TO JUDGE": "var(--dim)" };
const LOOKS_LIKE = { trading: "trading results", ab: "an A/B test", predictions: "model predictions",
                     data: "general data" };

const state = { kind: "auto", src: "upload", result: null, opt: {}, sample: false,
                code: null, codeId: null, ai: null, mlflow: null };

// ── tiny DOM helper ─────────────────────────────────────────────────────────
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "text") n.textContent = v;
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

function status(msg, error = false) {
  $("status").textContent = msg || "";
  $("status").className = error ? "error" : "";
}

function accessCode() {
  try { return localStorage.getItem("vision-code") || ""; } catch { return ""; }
}

async function call(path, body, isForm) {
  let r;
  const headers = isForm ? {} : { "Content-Type": "application/json" };
  if (accessCode()) headers["X-Vision-Code"] = accessCode();
  try {
    r = await fetch(API + path, { method: "POST", headers, body: isForm ? body : JSON.stringify(body) });
  } catch {
    throw new Error("Can't reach the VISION server - check your connection.");
  }
  const data = await r.json().catch(() => ({}));
  if (r.status === 401) { $("access").hidden = false; $("access-code").focus(); }
  if (!r.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : (r.status === 413 ? "That file is too big." : `Server error (${r.status}).`));
  }
  return data;
}

async function busy(button, label, fn) {
  const old = button.textContent;
  button.disabled = true; button.textContent = label;
  status("");
  try { await fn(); }
  catch (e) { status(e.message, true); }
  finally { button.disabled = false; button.textContent = old; }
}

// ── layout switching ────────────────────────────────────────────────────────
function setKind(kind) {
  state.kind = kind; state.result = null; state.opt = {}; state.mlflow = null;
  document.querySelectorAll("#kinds button").forEach((b) => b.classList.toggle("on", b.dataset.kind === kind));
  $("results-card").hidden = kind === "code";
  $("ml-files").hidden = kind !== "ml";
  $("drop-single").hidden = kind === "ml";
  $("sample").hidden = !["auto", "trading"].includes(kind);
  $("source").hidden = kind === "ml";
  if (kind === "ml") setSrc("upload");
  $("code-card").open = kind === "code" || $("code-card").open;
  $("code-title").firstChild.textContent = kind === "code" ? "The code / pipeline to check " : "Also check the code that produced this ";
  $("options").hidden = true;
  render();
}

function setSrc(src) {
  state.src = src; state.result = null; state.opt = {};
  document.querySelectorAll("#source button").forEach((b) => b.classList.toggle("on", b.dataset.src === src));
  $("upload-pane").hidden = src !== "upload";
  $("connect-pane").hidden = src !== "connect";
  $("options").hidden = true;
}

function showProvider() {
  const p = $("provider").value;
  document.querySelectorAll("#connect-pane .fields").forEach((f) => (f.hidden = f.dataset.provider !== p));
}

// ── options (shown after the first check, so pickers list the real columns) ─
function select(label, key, values, chosen, optional) {
  const s = el("select", { onchange: (e) => (state.opt[key] = e.target.value || null) },
    optional ? el("option", { value: "", text: "(none)" }) : null,
    values.map((v) => el("option", { value: v, text: v, selected: v === chosen })));
  state.opt[key] = chosen ?? (optional ? null : values[0]);
  return el("div", {}, el("label", { text: label }), s);
}

function number(label, key, value, attrs = {}) {
  state.opt[key] = state.opt[key] ?? value;
  return el("div", {}, el("label", { text: label }),
    el("input", { type: "number", value: state.opt[key], inputmode: "decimal", ...attrs,
                  oninput: (e) => (state.opt[key] = e.target.value === "" ? value : Number(e.target.value)) }));
}

function mlflowBlock(onRuns) {
  const host = el("input", { placeholder: "MLflow / Databricks URL", inputmode: "url", autocapitalize: "off" });
  const exp = el("input", { placeholder: "Experiment ID(s), comma-separated" });
  const tok = el("input", { type: "password", placeholder: "Token (if needed)", autocomplete: "off" });
  const btn = el("button", { class: "secondary", type: "button", text: "Pull runs" });
  const note = el("p", { class: "hint" });
  btn.addEventListener("click", () => busy(btn, "Pulling…", async () => {
    state.mlflow = await call("/api/mlflow", { host: host.value.trim(), experiment_ids: exp.value, token: tok.value });
    note.textContent = `${state.mlflow.runs} runs found.`;
    onRuns(state.mlflow);
  }));
  return el("details", {}, el("summary", { text: "Count the versions you tried from MLflow / Databricks" }),
    el("p", { class: "hint", text: "Pulls every run so 'versions tried' is counted, not guessed." }),
    host, exp, tok, btn, note);
}

function renderOptions() {
  const box = $("options");
  box.replaceChildren();
  const r = state.result;
  if (!r) { box.hidden = true; return; }
  const kind = r.kind, c = r.chosen || {}, cols = r.columns;
  const prev = { ...state.opt };
  state.opt = {};
  const keep = (k, v) => (prev[k] !== undefined ? prev[k] : v);
  box.append(el("h3", { text: state.kind === "auto" ? `Looks like ${LOOKS_LIKE[kind] || kind} - adjust if needed` : "Settings" }));

  if (kind === "trading") {
    box.append(el("div", { class: "row" }, select("% return column", "returns", cols, keep("returns", c.returns)),
                  select("Date column", "date", cols, keep("date", c.date), true)));
    state.opt.cost_bps = prev.cost_bps; state.opt.n_trials = prev.n_trials;
    const tries = number("Versions you tried", "n_trials", 1, { min: 1 });
    box.append(el("div", { class: "row" }, number("Cost per trade (bps)", "cost_bps", 5, { min: 0, step: "0.5" }), tries));
    box.append(mlflowBlock((m) => { state.opt.n_trials = m.runs; tries.querySelector("input").value = m.runs; }));
  } else if (kind === "ab") {
    box.append(el("div", { class: "row" }, select("Group column", "group", cols, keep("group", c.group)),
                  select("Outcome column", "outcome", cols, keep("outcome", c.outcome))));
    const guess = (c.groups || []).find((g) => ["control", "a", "0", "baseline"].includes(g.toLowerCase()));
    box.append(el("div", { class: "row" }, select("Time column", "time", cols, keep("time", c.time), true),
                  select("Control group", "control", c.groups || [], keep("control", guess))));
    state.opt.expected_split = prev.expected_split; state.opt.n_metrics = prev.n_metrics;
    box.append(el("div", { class: "row" }, number("Planned share in control (0-1)", "expected_split", 0.5, { step: "0.05", min: 0.01, max: 0.99 }),
                  number("Metrics you looked at", "n_metrics", 1, { min: 1 })));
  } else if (kind === "predictions") {
    box.append(el("div", { class: "row" }, select("Actual column", "y_true", cols, keep("y_true", c.y_true)),
                  select("Predicted column", "y_pred", cols, keep("y_pred", c.y_pred))));
  } else if (kind === "ml") {
    if (cols.train) box.append(el("div", { class: "row" }, select("Target column", "target", cols.train, keep("target", null)),
                                  select("Time column", "time", cols.train, keep("time", null), true)),
                               select("Entity / user ID column", "id", cols.train, keep("id", null), true));
    if (cols.predictions) box.append(el("div", { class: "row" },
      select("Actual column", "y_true", cols.predictions, keep("y_true", null)),
      select("Predicted column", "y_pred", cols.predictions, keep("y_pred", null))));
    const metricBox = el("div");
    box.append(mlflowBlock((m) => {
      metricBox.replaceChildren();
      const names = Object.keys(m.metrics);
      if (!names.length) { metricBox.append(el("p", { class: "hint", text: "Those runs have no metrics." })); return; }
      const sel = select("Metric that picked the winner (0-1)", "_metric", names, names[0]);
      const set = () => (state.opt.run_scores = m.metrics[state.opt._metric]);
      sel.querySelector("select").addEventListener("change", set); set();
      state.opt.higher_is_better = true;
      metricBox.append(sel, el("div", { class: "row" }, number("Test-set size", "n_test", 1000, { min: 1 }),
        el("label", { class: "check" }, el("input", { type: "checkbox", checked: true,
          onchange: (e) => (state.opt.higher_is_better = e.target.checked) }), "Higher is better")));
    }));
    box.append(metricBox);
  } else {
    box.append(el("p", { class: "hint", text: "No settings needed - every applicable integrity check runs." }));
  }
  box.hidden = false;
}

// ── running checks ──────────────────────────────────────────────────────────
function opts() {
  const o = {};
  for (const [k, v] of Object.entries(state.opt)) if (v !== null && v !== undefined && !k.startsWith("_")) o[k] = v;
  return o;
}

async function runResults() {
  const kind = state.kind;
  if (kind === "ml") {
    const fd = new FormData();
    for (const [field, id] of [["train", "ml-train"], ["test", "ml-test"], ["predictions", "ml-pred"]])
      if ($(id).files[0]) fd.append(field, $(id).files[0]);
    if (!fd.has("predictions") && !(fd.has("train") && fd.has("test")) && !opts().run_scores)
      throw new Error("Choose train + test tables, and/or a predictions table.");
    fd.append("options", JSON.stringify(opts()));
    state.result = await call("/api/check/ml", fd, true);
  } else if (state.src === "connect") {
    const p = $("provider").value;
    const credentials = p === "ibkr" ? { token: $("ibkr-token").value.trim(), query_id: $("ibkr-query").value.trim() }
      : p === "alpaca" ? { key_id: $("alpaca-id").value.trim(), secret: $("alpaca-secret").value.trim(), paper: $("alpaca-paper").checked }
      : { url: $("url").value.trim(), header: $("url-header").value.trim(), key: $("url-key").value, json_path: $("url-path").value.trim() };
    if (Object.values(credentials).some((v) => v === "") && p !== "url") throw new Error("Fill in the account details first.");
    if (p === "url" && !credentials.url) throw new Error("Enter the URL first.");
    status(p === "ibkr" ? "IBKR is building your statement - this can take ~30 seconds…" : "Connecting…");
    state.result = await call("/api/connect", { provider: p, kind: state.kind, options: opts(), credentials });
  } else {
    let file = $("file").files[0];
    if (!file && state.sample) {
      const blob = await (await fetch("sample.csv")).blob();
      file = new File([blob], "sample.csv", { type: "text/csv" });
    }
    if (!file) throw new Error("Choose a file first (or try the sample).");
    const fd = new FormData();
    fd.append("kind", state.kind);
    fd.append("options", JSON.stringify(opts()));
    fd.append("file", file);
    state.result = await call("/api/check", fd, true);
  }
  status("");
  renderOptions();
  await render();
}

async function runCode() {
  const files = [...$("code-files").files];
  const repo = $("repo").value.trim();
  if (!files.length && !repo) throw new Error("Choose code files or enter a GitHub repo.");
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  if (!files.length) { fd.append("repo", repo); fd.append("token", $("gh-token").value.trim()); }
  status(repo && !files.length ? "Downloading and reading the repo…" : "Reading the code…");
  const r = await call("/api/code", fd, true);
  state.code = r.report; state.codeId = r.code_id; state.ai = null;
  $("ai").hidden = false;
  status("");
  await render();
}

async function runAI() {
  if (!state.codeId) throw new Error("Audit the code first.");
  status("Claude is reading the code - large projects take a few minutes…");
  state.ai = await call("/api/ai-review", { code_id: state.codeId, anthropic_key: $("ai-key").value.trim() });
  status("");
  await render();
}

// ── verdict ─────────────────────────────────────────────────────────────────
async function render() {
  const out = $("verdict");
  const reports = [state.result && state.result.report, state.code, state.ai && state.ai.report].filter(Boolean);
  if (!reports.length) { out.replaceChildren(); return; }
  const rep = reports.length === 1 ? reports[0] : await call("/api/combine", { reports });
  const color = COLORS[rep.verdict] || "var(--dim)";
  const problems = rep.checks.filter((c) => c.status === "fail" || c.status === "warn")
    .sort((a, b) => (a.status === "fail" ? 0 : 1) - (b.status === "fail" ? 0 : 1));
  const passed = rep.checks.filter((c) => c.status === "pass");
  const skipped = rep.checks.filter((c) => c.status === "skip");

  out.replaceChildren(...[
    el("div", { class: "verdict", style: `border-color:${color}` },
      el("div", { class: "big", style: `color:${color}`, text: rep.verdict }),
      el("span", { class: "score", style: `color:${color};border-color:${color}`, text: rep.score }),
      el("div", { class: "sub", text: (rep.n ? `${rep.n} ${rep.unit} analyzed · ` : "") + rep.kind })),
    state.ai && state.ai.summary ? el("div", { class: "summary", text: "AI review: " + state.ai.summary }) : null,
    problems.length ? el("h2", { text: "What's wrong" }) : null,
    problems.map((c) => {
      const locs = (c.detail && c.detail.locations) || [];
      return el("div", { class: "problem" },
        el("div", { class: "name" }, el("span", { class: "mark", style: `color:${c.status === "fail" ? "var(--fake)" : "var(--warn)"}`,
          text: c.status === "fail" ? "✗" : "!" }), c.name),
        el("div", { class: "why", text: c.reason }),
        c.fix ? el("div", { class: "fix", text: "Fix: " + c.fix }) : null,
        locs.length > 1 ? el("details", {}, el("summary", { text: `All ${locs.length} places` }),
          el("pre", { text: locs.map((l) => `${l.file}:${l.line}  ${l.code || ""}`).join("\n") })) : null);
    }),
    !problems.length && passed.length ? el("div", { class: "summary",
      text: "Passed every check VISION could run. Still confirm it on fresh data before betting on it." }) : null,
    el("details", { class: "passed" }, el("summary", { text: `Passed (${passed.length}) · Couldn't run (${skipped.length})` }),
      el("ul", {}, passed.map((c) => el("li", { text: "✓ " + c.name })),
        skipped.map((c) => el("li", { text: `– ${c.name} — ${c.reason}` }))))].flat().filter(Boolean));
  out.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── wiring ──────────────────────────────────────────────────────────────────
document.querySelectorAll("#kinds button").forEach((b) => b.addEventListener("click", () => setKind(b.dataset.kind)));
document.querySelectorAll("#source button").forEach((b) => b.addEventListener("click", () => setSrc(b.dataset.src)));
$("provider").addEventListener("change", showProvider);
$("file").addEventListener("change", () => {
  const f = $("file").files[0];
  state.sample = false; state.result = null; state.opt = {}; $("options").hidden = true;
  $("file-name").textContent = f ? f.name : "Any layout - VISION figures it out";
  $("drop-single").classList.toggle("has", !!f);
});
for (const id of ["ml-train", "ml-test", "ml-pred"]) $(id).addEventListener("change", () => {
  const f = $(id).files[0];
  document.querySelector(`[data-for="${id}"]`).textContent = f ? f.name : "optional";
  $(id).closest(".drop").classList.toggle("has", !!f);
  state.result = null; $("options").hidden = true;
});
$("code-files").addEventListener("change", () => {
  const fs = [...$("code-files").files];
  $("code-names").textContent = fs.length ? fs.map((f) => f.name).join(", ") : "or use a GitHub repo below";
  $("code-files").closest(".drop").classList.toggle("has", fs.length > 0);
});
$("sample").addEventListener("click", () => {
  state.sample = true; $("file").value = "";
  $("file-name").textContent = "sample.csv (built-in example)";
  $("drop-single").classList.add("has");
  busy($("run"), "Checking…", runResults);
});
$("run").addEventListener("click", () => busy($("run"), "Checking…", runResults));
$("run-code").addEventListener("click", () => busy($("run-code"), "Auditing…", runCode));
$("run-ai").addEventListener("click", () => busy($("run-ai"), "Reviewing…", runAI));
showProvider();

if ("serviceWorker" in navigator && location.protocol.startsWith("http")) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

$("access-save").addEventListener("click", () => {
  try { localStorage.setItem("vision-code", $("access-code").value.trim()); } catch { /* private mode */ }
  $("access").hidden = true;
  status("Code saved - tap Check it again.");
});
