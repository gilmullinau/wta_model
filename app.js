// app.js — WTA Match Outcome predictor
// Loads TensorFlow.js (global tf), dataset from wta_data.csv, trains MLP model, visualizes metrics.

import { DataLoader } from "./data-loader.js";
import { ModelMLP } from "./gru.js";

const tf = window.tf; // Use global TensorFlow.js loaded via <script>
const LOG_MAX_LINES = 400;
const SCENARIO_YEAR = 2025;
const DEFAULT_HYPERPARAMS = {
  batchSize: 192,
  validationSplit: 0.2,
  hiddenUnits: [128, 64, 32],
  dropout: 0,
  starterEpochs: 2,
  starterSample: 900,
};

let loader = null;
let model = null;
let dataset = null;
let lossChart = null;
let cmChart = null;
let currentAutoVector = null;
let currentAutoPayload = null;

const SAVED_MODEL_KEY = "localstorage://wta-mlp-v2";

const els = {
  trainBtn: document.getElementById("trainBtn"),
  evalBtn: document.getElementById("evalBtn"),
  saveBtn: document.getElementById("saveBtn"),
  loadModelBtn: document.getElementById("loadModelBtn"),
  logs: document.getElementById("logs"),
  info: document.getElementById("info"),
  lossCanvas: document.getElementById("lossChart"),
  cmCanvas: document.getElementById("cmChart"),
  predictPanel: document.getElementById("predictPanel"),
  player1Select: document.getElementById("player1Select"),
  player2Select: document.getElementById("player2Select"),
  surfaceSelect: document.getElementById("surfaceSelect"),
  courtSelect: document.getElementById("courtSelect"),
  roundSelect: document.getElementById("roundSelect"),
  matchSummary: document.getElementById("matchSummary"),
  predictBtn: document.getElementById("predictBtn"),
  predictOut: document.getElementById("predictOut"),
  playerCards: document.getElementById("playerCards"),
  player1Card: document.getElementById("player1Card"),
  player2Card: document.getElementById("player2Card"),
  fileInput: document.getElementById("fileInput"),
  loadFileBtn: document.getElementById("loadFileBtn"),
  epochsInput: document.getElementById("epochsInput"),
  batchSizeInput: document.getElementById("batchSizeInput"),
  valSplitInput: document.getElementById("valSplitInput"),
  layer1Input: document.getElementById("layer1Units"),
  layer2Input: document.getElementById("layer2Units"),
  dropoutInput: document.getElementById("dropoutRate"),
  clearLogsBtn: document.getElementById("clearLogsBtn"),
  modelStatus: document.getElementById("modelStatus"),
  tennisBg: document.getElementById("tennisBg"),
  player1Pros: document.getElementById("player1Pros"),
  player1Cons: document.getElementById("player1Cons"),
  player2Pros: document.getElementById("player2Pros"),
  player2Cons: document.getElementById("player2Cons"),
  neutralInsights: document.getElementById("neutralInsights"),
  friendlyProgress: document.getElementById("friendlyProgress"),
  friendlySpinner: document.getElementById("friendlySpinner"),
  friendlyStatusText: document.getElementById("friendlyStatusText"),
};

const CATEGORY_FIELDS = [
  { key: "Surface", el: els.surfaceSelect, placeholder: "Select surface…" },
  { key: "Court", el: els.courtSelect, placeholder: "Select court…" },
  { key: "Round", el: els.roundSelect, placeholder: "Select round…" }
];

function log(msg) {
  const time = new Date().toLocaleTimeString();
  if (els.logs) {
    els.logs.textContent += `[${time}] ${msg}\n`;
    const lines = els.logs.textContent.split("\n");
    if (lines.length > LOG_MAX_LINES) {
      const trimmed = lines.slice(-LOG_MAX_LINES).join("\n");
      els.logs.textContent = trimmed.endsWith("\n") ? trimmed : `${trimmed}\n`;
    }
    els.logs.scrollTop = els.logs.scrollHeight;
  }
  console.log(`[${time}] ${msg}`);
}

function setModelStatus(text, options = {}) {
  if (els.modelStatus) {
    els.modelStatus.textContent = text;
  }
  if (els.friendlyStatusText) {
    els.friendlyStatusText.textContent = text;
  }
  if (els.friendlySpinner) {
    els.friendlySpinner.style.visibility = options.busy ? "visible" : "hidden";
  }
}

function featureArraysEqual(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function isModelCompatibleWithDataset(modelInstance) {
  if (!modelInstance || !dataset) return false;
  const expectedLen = dataset.featureNames?.length || 0;
  const loadedLen = Number.isFinite(modelInstance.inputDim)
    ? modelInstance.inputDim
    : (modelInstance.model?.inputs?.[0]?.shape?.[1] || 0);
  if (expectedLen !== loadedLen) return false;
  if (Array.isArray(modelInstance.featureNames) && modelInstance.featureNames.length) {
    return featureArraysEqual(modelInstance.featureNames, dataset.featureNames);
  }
  return true;
}

async function ensurePredictCompatibility() {
  if (!dataset) throw new Error("Dataset not loaded.");
  if (model && isModelCompatibleWithDataset(model)) return true;

  await purgeSavedModel("schema mismatch before prediction");
  await ensureModelReady();

  if (!model || !isModelCompatibleWithDataset(model)) {
    throw new Error("Model schema still mismatched after refresh. Please reload the page.");
  }
  return true;
}

async function purgeSavedModel(reason = "") {
  try {
    await tf.io.removeModel(SAVED_MODEL_KEY);
    localStorage.removeItem("wta-mlp-v2-meta");
    if (reason) log(`Cleared saved model: ${reason}`); else log("Cleared saved model.");
  } catch (err) {
    console.warn("Failed to purge saved model", err);
  }
}

function enableTraining(enabled) {
  if (els.trainBtn) els.trainBtn.disabled = !enabled;
  if (els.evalBtn) els.evalBtn.disabled = !enabled || !model;
  if (els.saveBtn) els.saveBtn.disabled = !enabled || !model;
}

function showPredictPanel(show) {
  els.predictPanel.style.display = show ? "block" : "none";
  if (!show) {
    resetAutoPredictPanel(`Select two players to build a ${SCENARIO_YEAR} matchup from the dataset.`);
  } else {
    updateAutoPreview();
  }
}

async function parseAndInit(text) {
  try {
    disposeDataset();
    if (model) {
      model.dispose();
      model = null;
    }
    if (lossChart) { lossChart.destroy(); lossChart = null; }
    if (cmChart) { cmChart.destroy(); cmChart = null; }
    loader = new DataLoader();
    dataset = await loader.loadCSVText(text);
    els.info.textContent = "Dataset loaded. Getting the model ready for picks…";
    log("Dataset loaded successfully.");
    enableTraining(false);
    buildPredictForm();
    if (els.saveBtn) els.saveBtn.disabled = true;
    showPredictPanel(false);
    await ensureModelReady();
  } catch (err) {
    console.error(err);
    els.info.textContent = "We hit a snag loading the data. Please try again.";
    log(`Dataset error: ${err.message}`);
    enableTraining(false);
  }
}

async function autoLoadCSV() {
  const url = `./wta_data.csv?v=${Date.now()}`;
  try {
    console.log("🔍 Fetching CSV from:", url);
    const res = await fetch(url, { cache: "no-store" });
    console.log("✅ HTTP status:", res.status);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    console.log("📄 CSV length:", text.length);
    console.log("📄 First 200 chars:", text.slice(0, 200));

    if (!text || text.trim().length === 0) throw new Error("CSV is empty");
    await parseAndInit(text);
  } catch (err) {
    console.error("❌ Auto-load failed:", err);
    log(`Auto-load failed: ${err.message}`);
    els.info.textContent = "We couldn't fetch the data. Please refresh to try again.";
  }
}

async function ensureModelReady() {
  if (!dataset) return;
  try {
    setModelStatus("Model: loading saved neural net…", { busy: true });
    const m = new ModelMLP(dataset.featureNames.length, {
      featureNames: dataset.featureNames,
      featureIndexMap: dataset.featureIndexMap,
    });
    await m.load();
    if (!isModelCompatibleWithDataset(m)) {
      const expected = dataset.featureNames.length;
      const found = m?.inputDim ?? "unknown";
      await purgeSavedModel(`schema mismatch (expected ${expected} features, found ${found})`);
      m.dispose();
      throw new Error("Saved model incompatible with current dataset schema.");
    }
    m.inputDim = dataset.featureNames.length;
    m.featureNames = dataset.featureNames.slice();
    m.featureIndexMap = { ...dataset.featureIndexMap };
    model = m;
    log("Model loaded from browser storage.");
    enableTraining(true);
    showPredictPanel(true);
    setModelStatus("Model: ready — pick two players", { busy: false });
    els.info.textContent = "Ready to predict. Pick two players below.";
    return;
  } catch (err) {
    console.warn("Starter model missing, training a fresh one.", err?.message);
  }

  try {
    setModelStatus("Model: quick-start training…", { busy: true });
    await trainModel({
      silent: true,
      autoSave: true,
      label: "Starter training",
      quickStarter: true,
    });
    setModelStatus("Model: ready — pick two players", { busy: false });
    els.info.textContent = "Quick-start model loaded. Pick two players below.";
  } catch (err) {
    console.error(err);
    log(`Starter training failed: ${err.message}`);
    setModelStatus("Model: needs training", { busy: false });
    els.info.textContent = "Model unavailable. Please reload the page.";
    enableTraining(true);
  }
}

function buildPredictForm() {
  if (!loader || !dataset) return;
  const players = loader.getPlayerNames();
  const player1Options = [
    "<option value=\"\">Select…</option>",
    ...players.map((p) => {
      const safe = escapeHtml(p);
      return `<option value="${safe}">${safe}</option>`;
    })
  ];
  els.player1Select.innerHTML = player1Options.join("");
  els.player1Select.value = "";
  setPlayer2Placeholder("Select Player 1 first…");
  buildCategoryControls();
  resetAutoPredictPanel(`Select two players to build a ${SCENARIO_YEAR} matchup from the dataset.`);
}

function resetAutoPredictPanel(message) {
  els.matchSummary.textContent = message;
  els.predictOut.textContent = "";
  renderPlayerCards(null);
  setInsightList(els.player1Pros, [], "Waiting for Player 1");
  setInsightList(els.player1Cons, [], "—");
  setInsightList(els.player2Pros, [], "Waiting for Player 2");
  setInsightList(els.player2Cons, [], "—");
  setInsightList(els.neutralInsights, [], "Pick players to see the matchup story.");
  currentAutoVector = null;
  currentAutoPayload = null;
  els.predictBtn.disabled = true;
  resetCategoryControls();
}

function setPlayer2Placeholder(text) {
  els.player2Select.innerHTML = `<option value="">${escapeHtml(text)}</option>`;
  els.player2Select.value = "";
  els.player2Select.disabled = true;
}

function buildCategoryControls() {
  if (!loader) {
    CATEGORY_FIELDS.forEach(({ el, placeholder }) => {
      if (el) {
        el.innerHTML = `<option value="">${escapeHtml(placeholder)}</option>`;
        el.value = "";
        el.disabled = true;
      }
    });
    return;
  }
  CATEGORY_FIELDS.forEach(({ key, el, placeholder }) => {
    if (!el) return;
    const options = loader.getCategoryOptions(key);
    const optionHtml = [
      `<option value="">${escapeHtml(placeholder)}</option>`,
      ...options.map((val) => {
        const safe = escapeHtml(val);
        return `<option value="${safe}">${safe}</option>`;
      })
    ];
    el.innerHTML = optionHtml.join("");
    el.value = "";
    el.disabled = true;
  });
}

function resetCategoryControls() {
  CATEGORY_FIELDS.forEach(({ el }) => {
    if (!el) return;
    el.value = "";
    el.disabled = true;
  });
}

function applyCategoryDefaultsFromPayload(payload) {
  CATEGORY_FIELDS.forEach(({ key, el }) => {
    if (!el) return;
    el.disabled = false;
    const candidate = payload.categorical?.[key] ?? "";
    const resolved = setSelectValue(el, candidate);
    payload.categorical[key] = resolved;
    if (payload.vectorInput) payload.vectorInput[key] = resolved;
    if (currentAutoVector) currentAutoVector[key] = resolved;
  });
}

function setSelectValue(selectEl, candidate) {
  const safe = (candidate ?? "").toString();
  const options = Array.from(selectEl?.options || []);
  if (safe && options.some((opt) => opt.value === safe)) {
    selectEl.value = safe;
  } else {
    selectEl.value = "";
  }
  return selectEl.value || "";
}

function handleCategorySelectChange(column) {
  if (!currentAutoPayload || !currentAutoVector) return;
  const field = CATEGORY_FIELDS.find((cfg) => cfg.key === column);
  if (!field || !field.el) return;
  const value = field.el.value || "";
  currentAutoPayload.categorical[column] = value;
  if (currentAutoPayload.vectorInput) currentAutoPayload.vectorInput[column] = value;
  currentAutoVector[column] = value;
  renderFriendlyInsights(currentAutoPayload);
}

function populatePlayer2Options(player1) {
  const opponents = loader.getOpponentsFor(player1);
  const prev = els.player2Select.value;
  if (!opponents || opponents.length === 0) {
    setPlayer2Placeholder("No opponents available");
    return { opponents: [], preserved: false };
  }
  const optionHtml = [
    "<option value=\"\">Select…</option>",
    ...opponents.map((name) => {
      const safe = escapeHtml(name);
      return `<option value="${safe}">${safe}</option>`;
    })
  ];
  els.player2Select.innerHTML = optionHtml.join("");
  els.player2Select.disabled = false;
  if (prev && opponents.includes(prev)) {
    els.player2Select.value = prev;
    return { opponents, preserved: true };
  }
  els.player2Select.value = "";
  return { opponents, preserved: false };
}

function escapeHtml(str) {
  return (str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function setInsightList(el, items, placeholder = "") {
  if (!el) return;
  if (!items || items.length === 0) {
    el.innerHTML = placeholder ? `<li>${escapeHtml(placeholder)}</li>` : "";
    return;
  }
  el.innerHTML = items.map((text) => `<li>${escapeHtml(text)}</li>`).join("");
}

function handlePlayer1Change() {
  if (!loader) return;
  const player1 = els.player1Select.value;
  if (!player1) {
    setPlayer2Placeholder("Select Player 1 first…");
    resetAutoPredictPanel(`Select two players to build a ${SCENARIO_YEAR} matchup from the dataset.`);
    return;
  }

  const { opponents, preserved } = populatePlayer2Options(player1);
  if (opponents.length === 0) {
    resetAutoPredictPanel(`No recorded opponents for ${player1} in the dataset.`);
    return;
  }

  if (preserved && els.player2Select.value) {
    updateAutoPreview();
  } else {
    resetAutoPredictPanel(`Select an opponent for ${player1} to pull their latest matchup stats.`);
  }
}

function updateAutoPreview() {
  if (!loader) return;
  const player1 = els.player1Select.value;
  const player2 = els.player2Select.value;
  if (!player1) {
    setPlayer2Placeholder("Select Player 1 first…");
    resetAutoPredictPanel(`Select two players to build a ${SCENARIO_YEAR} matchup from the dataset.`);
    return;
  }
  if (!player2) {
    resetAutoPredictPanel(`Select an opponent for ${player1} to pull their latest matchup stats.`);
    return;
  }
  if (player1 === player2) {
    resetAutoPredictPanel("Choose two different players to build a matchup.");
    return;
  }
  const payload = loader.getLatestAutoFeatures(player1, player2);
  if (!payload) {
    resetAutoPredictPanel("No matchup with these players was found in the dataset. Try another pairing.");
    return;
  }
  currentAutoPayload = payload;
  currentAutoVector = { ...payload.vectorInput };
  applyCategoryDefaultsFromPayload(currentAutoPayload);
  renderFriendlyInsights(currentAutoPayload);
  els.matchSummary.textContent = describeMatchSummary(currentAutoPayload);
  if (!model) {
    els.predictBtn.disabled = true;
    els.predictOut.textContent = "Train or load a model to enable prediction.";
  } else {
    els.predictBtn.disabled = false;
    els.predictOut.textContent = "";
  }
}

function renderFriendlyInsights(payload) {
  renderPlayerCards(payload);
  if (!payload) return;
  const { pros1, cons1, pros2, cons2, neutral } = buildInsights(payload);
  setInsightList(els.player1Pros, pros1, "Waiting for Player 1");
  setInsightList(els.player1Cons, cons1, "No obvious risks");
  setInsightList(els.player2Pros, pros2, "Waiting for Player 2");
  setInsightList(els.player2Cons, cons2, "No obvious risks");
  setInsightList(els.neutralInsights, neutral, "Pick players to see the matchup story.");
}

function formatFeatureValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (key === "year") return `${value} (scenario year)`;
  if (key === "age") return `${Number(value).toFixed(1)} yrs`;
  if (key === "rank_diff") {
    const p1 = currentAutoPayload?.players?.player1 || "Player 1";
    const p2 = currentAutoPayload?.players?.player2 || "Player 2";
    return `${Number(value).toFixed(0)} (${p2} rank − ${p1} rank)`;
  }
  if (key === "pts_diff") {
    const p1 = currentAutoPayload?.players?.player1 || "Player 1";
    const p2 = currentAutoPayload?.players?.player2 || "Player 2";
    return `${Number(value).toFixed(0)} (${p1} pts − ${p2} pts from last meeting)`;
  }
  if (key === "last_winner") {
    const label = currentAutoPayload?.players?.player1 || "Player 1";
    return `${value} (${value === 1 ? `${label} won last` : `${label} did not win last`})`;
  }
  if (key === "recent5" || key === "recent10") {
    return `${formatPercent(value)} win rate`;
  }
  if (key?.startsWith("fatigue")) {
    return `${Number(value).toFixed(1)} workload index`;
  }
  const abs = Math.abs(value);
  const decimals = abs >= 100 ? 1 : 3;
  return Number(value).toFixed(decimals);
}

function formatAgeGroup(value) {
  if (!value) return "—";
  const map = {
    lt20: "<20",
    "20_24": "20–24",
    "25_29": "25–29",
    "30_34": "30–34",
    "35_plus": "35+",
  };
  return map[value] || escapeHtml(value.toString());
}

function buildInsights(payload) {
  const numeric = payload?.numeric || {};
  const ageGroups = payload?.ageGroups || {};
  const players = payload?.players || {};
  const p1 = players.player1 || "Player 1";
  const p2 = players.player2 || "Player 2";

  const pros1 = [], pros2 = [], cons1 = [], cons2 = [], neutral = [];

  const rankDiff = numeric.rank_diff;
  if (isFiniteNumber(rankDiff)) {
    if (rankDiff < -2) {
      pros1.push(`${p1} is ranked ${Math.abs(rankDiff).toFixed(0)} spots higher than ${p2}.`);
      cons2.push(`${p2} trails in ranking for now.`);
    } else if (rankDiff > 2) {
      pros2.push(`${p2} holds a ranking edge of ${Math.abs(rankDiff).toFixed(0)} spots.`);
      cons1.push(`${p1} will need to punch above ranking.`);
    } else neutral.push("Rankings are very close on paper.");
  }

  const pointsDiff = numeric.pts_diff;
  if (isFiniteNumber(pointsDiff)) {
    const edge = Math.abs(pointsDiff);
    if (edge >= 25) {
      const target = pointsDiff >= 0 ? pros1 : pros2;
      target.push(`Recent points tilt toward ${pointsDiff >= 0 ? p1 : p2} by about ${edge.toFixed(0)}.`);
    }
  }

  if (numeric.last_winner === 1) pros1.push(`${p1} won the last meeting.`);
  if (numeric.last_winner === 0) pros2.push(`${p2} won the last meeting.`);

  if (isFiniteNumber(numeric.h2h_advantage)) {
    if (numeric.h2h_advantage > 0) pros1.push(`${p1} leads the head-to-head record.`);
    else if (numeric.h2h_advantage < 0) pros2.push(`${p2} leads the head-to-head record.`);
  }

  const win5_1 = numeric.recent_win_rate_5_1;
  const win5_2 = numeric.recent_win_rate_5_2;
  if (isFiniteNumber(win5_1) && isFiniteNumber(win5_2)) {
    const diff = win5_1 - win5_2;
    if (Math.abs(diff) >= 0.05) {
      const target = diff > 0 ? pros1 : pros2;
      target.push(`${diff > 0 ? p1 : p2} has the hotter 5-match win rate (${formatPercent(Math.max(win5_1, win5_2))}).`);
    }
  }

  const win10_1 = numeric.recent_win_rate_10_1;
  const win10_2 = numeric.recent_win_rate_10_2;
  if (isFiniteNumber(win10_1) && isFiniteNumber(win10_2)) {
    const diff = win10_1 - win10_2;
    if (Math.abs(diff) >= 0.05) {
      const target = diff > 0 ? pros1 : pros2;
      target.push(`Over 10 matches, ${diff > 0 ? p1 : p2} has steadier results (${formatPercent(Math.max(win10_1, win10_2))}).`);
    }
  }

  const streakValue1 = numeric.streak_value_1;
  const streakValue2 = numeric.streak_value_2;
  if (isFiniteNumber(streakValue1) && isFiniteNumber(streakValue2) && Math.abs(streakValue1 - streakValue2) >= 1) {
    const target = streakValue1 > streakValue2 ? pros1 : pros2;
    target.push(`${streakValue1 > streakValue2 ? p1 : p2} comes in on the stronger streak.`);
  }

  const fatigue1 = average([numeric.fatigue_7d_1, numeric.fatigue_14d_1, numeric.fatigue_30d_1]);
  const fatigue2 = average([numeric.fatigue_7d_2, numeric.fatigue_14d_2, numeric.fatigue_30d_2]);
  if (isFiniteNumber(fatigue1) && isFiniteNumber(fatigue2)) {
    const diff = fatigue2 - fatigue1;
    if (diff > 1) {
      pros1.push(`${p1} looks fresher based on recent workload.`);
      cons2.push(`${p2} has logged more minutes recently.`);
    } else if (diff < -1) {
      pros2.push(`${p2} looks fresher based on recent workload.`);
      cons1.push(`${p1} has logged more minutes recently.`);
    }
  }

  if (isFiniteNumber(numeric.surface_trend_1) && isFiniteNumber(numeric.surface_trend_2)) {
    const diff = numeric.surface_trend_1 - numeric.surface_trend_2;
    if (diff > 0.1) pros1.push(`${p1} has better momentum on this surface.`);
    else if (diff < -0.1) pros2.push(`${p2} has better momentum on this surface.`);
    else neutral.push("Surface trends are evenly matched.");
  }

  if (ageGroups.age_group_1 || ageGroups.age_group_2) {
    neutral.push(`${p1} age group: ${formatAgeGroup(ageGroups.age_group_1)}; ${p2} age group: ${formatAgeGroup(ageGroups.age_group_2)}.`);
  }

  const oddDiff = numeric.odd_diff;
  if (isFiniteNumber(oddDiff) && Math.abs(oddDiff) > 0.05) {
    const fav = oddDiff < 0 ? p1 : p2;
    neutral.push(`${fav} entered the last matchup as the favored player.`);
  }

  return { pros1, cons1, pros2, cons2, neutral };
}

function renderPlayerCards(payload) {
  if (!els.playerCards || !els.player1Card || !els.player2Card) return;
  if (!payload) {
    els.player1Card.innerHTML = "<div class=\"small muted\">Player 1 stats will appear here.</div>";
    els.player2Card.innerHTML = "<div class=\"small muted\">Player 2 stats will appear here.</div>";
    return;
  }
  const { players = {}, playerFeatures = {} } = payload;
  const renderCard = (cardEl, nameKey) => {
    const name = players[nameKey] || nameKey;
    const stats = playerFeatures[name] || {};
    const rows = [
      `<div class="card-title">${escapeHtml(name || "—")}</div>`,
      `<div class="stat-line"><span>Age</span><span>${formatFeatureValue("age", stats.age)}</span></div>`,
      `<div class="stat-line"><span>Age group</span><span>${formatAgeGroup(stats.ageGroup)}</span></div>`,
      `<div class="stat-line"><span>Form streak</span><span>${formatFeatureValue("streak", stats.streak)}</span></div>`,
      `<div class="stat-line"><span>Momentum</span><span>${formatFeatureValue("streak_value", stats.streakValue)}</span></div>`,
      `<div class="stat-line"><span>Win rate (5)</span><span>${formatFeatureValue("recent5", stats.recent5)}</span></div>`,
      `<div class="stat-line"><span>Win rate (10)</span><span>${formatFeatureValue("recent10", stats.recent10)}</span></div>`,
      `<div class="stat-line"><span>Fatigue 7d</span><span>${formatFeatureValue("fatigue7", stats.fatigue7)}</span></div>`,
      `<div class="stat-line"><span>Fatigue 14d</span><span>${formatFeatureValue("fatigue14", stats.fatigue14)}</span></div>`,
      `<div class="stat-line"><span>Fatigue 30d</span><span>${formatFeatureValue("fatigue30", stats.fatigue30)}</span></div>`,
      `<div class="stat-line"><span>Surface trend</span><span>${formatFeatureValue("surface_trend", stats.surfaceTrend)}</span></div>`,
    ];
    cardEl.innerHTML = rows.join("");
  };
  renderCard(els.player1Card, "player1");
  renderCard(els.player2Card, "player2");
}

function describeMatchSummary(payload) {
  const { datasetMatch, players, playerSnapshots } = payload;
  const segments = [];
  if (players?.player1 && players?.player2) {
    segments.push(`${players.player1} vs ${players.player2} set for ${SCENARIO_YEAR}.`);
  } else {
    segments.push(`Scenario year fixed to ${SCENARIO_YEAR}.`);
  }

  if (datasetMatch?.player1 || datasetMatch?.player2 || datasetMatch?.date) {
    let line = "Latest recorded meeting";
    if (datasetMatch.player1 && datasetMatch.player2) {
      line += `: ${datasetMatch.player1} vs ${datasetMatch.player2}`;
    }
    if (datasetMatch.date) {
      line += ` on ${datasetMatch.date}`;
    }
    if (datasetMatch.winner) {
      line += ` — winner ${datasetMatch.winner}`;
    }
    if (datasetMatch.score) {
      const cleanScore = datasetMatch.score.replace(/\s+/g, " ").trim();
      if (cleanScore.length > 0) {
        line += ` (${cleanScore})`;
      }
    }
    if (datasetMatch.orientation === "reverse") {
      line += " (order flipped to match your selection)";
    }
    segments.push(`${line}.`);
  } else {
    segments.push("Latest head-to-head record pulled from dataset.");
  }

  if (players?.player1 && players?.player2) {
    const snapshotLine = buildSnapshotLine(
      players.player1,
      playerSnapshots?.[players.player1],
      players.player2,
      playerSnapshots?.[players.player2]
    );
    if (snapshotLine) segments.push(snapshotLine);
  }

  segments.push("Surface, court, and round default to the most recent clash — adjust them for your scenario.");
  return segments.join(" ");
}

function buildSnapshotLine(player1Name, player1Snap, player2Name, player2Snap) {
  const details = [];
  const left = formatSnapshotSummary(player1Name, player1Snap);
  const right = formatSnapshotSummary(player2Name, player2Snap);
  if (left) details.push(left);
  if (right) details.push(right);
  if (details.length === 0) return null;
  return `Latest rankings — ${details.join("; ")}.`;
}

function formatSnapshotSummary(name, snap) {
  if (!name) return null;
  if (!snap || (!isFiniteNumber(snap.rank) && !isFiniteNumber(snap.pts))) {
    return `${name}: no recent ranking data`;
  }
  const pieces = [];
  if (isFiniteNumber(snap.rank)) {
    pieces.push(`rank ${Math.round(snap.rank)}`);
  }
  if (isFiniteNumber(snap.pts)) {
    pieces.push(`${Math.round(snap.pts).toLocaleString()} pts`);
  }
  let text = `${name}: ${pieces.join(", ")}`;
  if (snap.date) {
    text += ` (as of ${snap.date})`;
  }
  return text;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatPercent(value) {
  if (!isFiniteNumber(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

function average(values = []) {
  const nums = values.filter(isFiniteNumber);
  if (nums.length === 0) return NaN;
  return nums.reduce((sum, v) => sum + v, 0) / nums.length;
}

async function trainModel(eventOrOpts) {
  const isEvent = eventOrOpts && typeof eventOrOpts.preventDefault === "function";
  if (isEvent) eventOrOpts.preventDefault();
  const opts = isEvent ? {} : (eventOrOpts || {});
  const silent = Boolean(opts.silent);
  const label = opts.label || "Training";
  const quickStarter = Boolean(opts.quickStarter);
  if (!dataset) {
    if (!silent) alert("Dataset not loaded yet.");
    return;
  }
  if (model) model.dispose();
  const hyper = readHyperparameters();
  model = new ModelMLP(dataset.featureNames.length, {
    ...hyper.architecture,
    featureNames: dataset.featureNames,
    featureIndexMap: dataset.featureIndexMap,
  });
  model.build();
  model.inputDim = dataset.featureNames.length;
  model.featureNames = dataset.featureNames.slice();
  model.featureIndexMap = { ...dataset.featureIndexMap };
  log(`${label} started${quickStarter ? " (quick sample)" : ""}...`);
  setModelStatus(`${label}: running`);
  const losses = [], valAcc = [];
  enableTraining(false);
  const cleanup = [];
  try {
    const sampleSize = quickStarter
      ? Math.min(DEFAULT_HYPERPARAMS.starterSample, dataset.X_train.shape[0])
      : dataset.X_train.shape[0];
    const trainXs = quickStarter
      ? tf.tidy(() => {
          const t = dataset.X_train.slice([0, 0], [sampleSize, dataset.X_train.shape[1]]);
          cleanup.push(t);
          return t;
        })
      : dataset.X_train;
    const trainYs = quickStarter
      ? tf.tidy(() => {
          const t = dataset.y_train.slice([0, 0], [sampleSize, 1]);
          cleanup.push(t);
          return t;
        })
      : dataset.y_train;

    await model.train(trainXs, trainYs, {
      epochs: quickStarter ? DEFAULT_HYPERPARAMS.starterEpochs : hyper.training.epochs,
      batchSize: quickStarter ? Math.min(DEFAULT_HYPERPARAMS.batchSize, sampleSize) : hyper.training.batchSize,
      validationSplit: quickStarter ? 0.1 : hyper.training.validationSplit,
      onEpochEnd: (epoch, logs) => {
        const val = logs.val_acc ?? logs.val_accuracy ?? 0;
        log(`Epoch ${epoch + 1}: loss=${Number(logs.loss).toFixed(4)} val_acc=${Number(val).toFixed(4)}`);
        losses.push(Number(logs.loss));
        valAcc.push(Number(val));
        drawLossChart(losses, valAcc);
      }
    });

    log(`${label} complete.`);
    if (els.saveBtn) els.saveBtn.disabled = false;
    if (els.evalBtn) els.evalBtn.disabled = false;
    showPredictPanel(true);
    setModelStatus("Model: ready");
    if (opts.autoSave) {
      await model.save();
      log("Starter model saved to browser storage.");
    }
    if (opts.onComplete) opts.onComplete();
  } catch (err) {
    log(`Training failed: ${err.message}`);
    if (!silent) alert(err.message);
    setModelStatus("Model: training failed");
    throw err;
  } finally {
    cleanup?.forEach?.((t) => t.dispose?.());
    enableTraining(true);
  }
}

async function evaluateModel() {
  if (!dataset || !model) return alert("Train the model first.");
  log("Evaluating on test set...");
  const { loss, acc, metricAcc } = await model.evaluate(dataset.X_test, dataset.y_test);
  log(`Test Loss=${loss.toFixed(4)} | Accuracy=${acc.toFixed(4)} (tf: ${metricAcc.toFixed(4)})`);
  const cm = await model.confusionMatrix(dataset.X_test, dataset.y_test);
  drawConfusionMatrix(cm);
}

function drawLossChart(losses, valAcc) {
  const ctx = els.lossCanvas?.getContext?.("2d");
  if (!ctx) return;
  if (lossChart) lossChart.destroy();
  lossChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: losses.map((_, i) => `E${i + 1}`),
      datasets: [
        { label: "Loss", data: losses, borderColor: "#6aa8ff", tension: 0.2 },
        { label: "Val Accuracy", data: valAcc, borderColor: "#50fa7b", tension: 0.2 }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false }
  });
}

function drawConfusionMatrix({ tp, tn, fp, fn }) {
  const ctx = els.cmCanvas?.getContext?.("2d");
  if (!ctx) return;
  if (cmChart) cmChart.destroy();
  cmChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Actual 0", "Actual 1"],
      datasets: [
        { label: "Pred 0", data: [tn, fn], backgroundColor: "#6aa8ff" },
        { label: "Pred 1", data: [fp, tp], backgroundColor: "#ff6384" }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: true } }
    }
  });
}

function initTennisBackground() {
  if (!els.tennisBg) return;
  const count = 18;
  for (let i = 0; i < count; i++) {
    const ball = document.createElement("span");
    ball.className = "tennis-ball";
    ball.style.left = `${Math.random() * 100}%`;
    ball.style.animationDelay = `${Math.random() * 6}s`;
    ball.style.setProperty("--duration", `${6 + Math.random() * 6}s`);
    ball.style.setProperty("--scale", `${0.6 + Math.random() * 0.8}`);
    els.tennisBg.appendChild(ball);
  }
}

async function handlePredict(e) {
  e.preventDefault();
  if (!loader || !dataset) return alert("Load data first.");
  if (!currentAutoVector) {
    alert("Select two players with available matchup data first.");
    return;
  }
  try {
    await ensurePredictCompatibility();
    const vec = loader.vectorizeForPredict(currentAutoVector);
    const x = tf.tensor2d([Array.from(vec)], [1, vec.length], "float32");
    const yProb = model.predictProba(x);
    const prob = (await yProb.data())[0];
    const pred = prob >= 0.5 ? 1 : 0;
    const player1 = currentAutoPayload?.players?.player1 || "Player 1";
    const player2 = currentAutoPayload?.players?.player2 || "Player 2";
    const outcome = pred === 1 ? `${player1} favored` : `${player2} favored`;
    const pct1 = (prob * 100).toFixed(1);
    const pct2 = (100 - prob * 100).toFixed(1);
    els.predictOut.textContent = `${outcome} — win chance ${pct1}% for ${player1} / ${pct2}% for ${player2}`;
    x.dispose(); yProb.dispose();
  } catch (err) {
    log(`Prediction failed: ${err.message}`);
    alert(err.message);
  }
}

// Buttons (guarded for hidden technical controls)
if (els.trainBtn) els.trainBtn.addEventListener("click", trainModel);
if (els.evalBtn) els.evalBtn.addEventListener("click", evaluateModel);
if (els.saveBtn) {
  els.saveBtn.addEventListener("click", async () => {
    if (model) { await model.save(); log("Model saved to browser storage."); }
  });
}
if (els.loadModelBtn) {
  els.loadModelBtn.addEventListener("click", async () => {
    try {
      setModelStatus("Model: loading saved neural net…");
      const m = new ModelMLP(dataset ? dataset.featureNames.length : 0, {
        featureNames: dataset?.featureNames || [],
        featureIndexMap: dataset?.featureIndexMap || {},
      });
      await m.load();
      if (dataset && !isModelCompatibleWithDataset(m)) {
        const expected = dataset.featureNames.length;
        const found = m?.inputDim ?? "unknown";
        await purgeSavedModel(`schema mismatch (expected ${expected} features, found ${found})`);
        m.dispose();
        throw new Error("Saved model incompatible with current dataset schema.");
      }
      model = m;
      log("Model loaded from browser storage.");
      showPredictPanel(true);
      enableTraining(Boolean(dataset));
      setModelStatus("Model: ready (restored)");
      if (!dataset || !loader) {
        log("Load a dataset to enable predictions with the restored model.");
      }
    } catch {
      alert("No saved model found or load failed.");
      setModelStatus("Model: needs training");
    }
  });
}
CATEGORY_FIELDS.forEach(({ key, el }) => {
  if (!el) return;
  el.addEventListener("change", () => handleCategorySelectChange(key));
});
els.player1Select.addEventListener("change", handlePlayer1Change);
els.player2Select.addEventListener("change", updateAutoPreview);
els.predictBtn.addEventListener("click", handlePredict);
if (els.loadFileBtn) els.loadFileBtn.addEventListener("click", handleManualFileLoad);
if (els.clearLogsBtn && els.logs) {
  els.clearLogsBtn.addEventListener("click", () => {
    els.logs.textContent = "";
  });
}

// Init
console.log("🚀 App initialized — calling autoLoadCSV()");
enableTraining(false);
buildCategoryControls();
showPredictPanel(false);
initTennisBackground();
autoLoadCSV();
console.log("✅ autoLoadCSV() call placed after init");

function readHyperparameters() {
  const epochValue = els.epochsInput?.value ?? DEFAULT_HYPERPARAMS.epochs ?? 6;
  const epochs = clampInt(epochValue, 1, 200, 6);
  return {
    training: {
      epochs,
      batchSize: DEFAULT_HYPERPARAMS.batchSize,
      validationSplit: DEFAULT_HYPERPARAMS.validationSplit,
    },
    architecture: {
      hiddenUnits: DEFAULT_HYPERPARAMS.hiddenUnits.slice(),
      dropout: DEFAULT_HYPERPARAMS.dropout,
    }
  };
}

function clampInt(value, min, max, fallback) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

async function handleManualFileLoad() {
  if (!els.fileInput || !els.fileInput.files || els.fileInput.files.length === 0) {
    return alert("Select a CSV file first.");
  }
  const file = els.fileInput.files[0];
  try {
    const text = await file.text();
    log(`Manual CSV load: ${file.name} (${text.length} chars)`);
    await parseAndInit(text);
  } catch (err) {
    log(`Manual load failed: ${err.message}`);
    alert(err.message);
  }
}

function disposeDataset() {
  if (!dataset) return;
  try {
    dataset.X_train?.dispose();
    dataset.y_train?.dispose();
    dataset.X_test?.dispose();
    dataset.y_test?.dispose();
  } catch (err) {
    console.warn("Failed to dispose dataset tensors", err);
  }
  dataset = null;
}
