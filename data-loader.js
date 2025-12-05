// data-loader.js
// Reads CSV text, preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// stratified split train/test, returns tf.Tensors and feature metadata.
const tf = window.tf;
const SCENARIO_YEAR = 2025;
const AGE_GROUPS = [
  { label: "lt20", min: -Infinity, max: 20 },
  { label: "20_24", min: 20, max: 25 },
  { label: "25_29", min: 25, max: 30 },
  { label: "30_34", min: 30, max: 35 },
  { label: "35_plus", min: 35, max: Infinity },
];

export class DataLoader {
  constructor() {
    this.numericCols = [
      // Static differentials
      "rank_diff", "pts_diff", "odd_diff", "h2h_advantage", "last_winner", "surface_winrate_adv",
      // Age
      "age_1", "age_2",
      // Form / streaks
      "streak_1", "streak_2", "streak_value_1", "streak_value_2",
      "recent_win_rate_5_1", "recent_win_rate_5_2", "recent_win_rate_10_1", "recent_win_rate_10_2",
      // Fatigue
      "fatigue_7d_1", "fatigue_7d_2", "fatigue_14d_1", "fatigue_14d_2", "fatigue_30d_1", "fatigue_30d_2",
      // Surface
      "surface_trend_1", "surface_trend_2",
    ];
    this.categoricalCols = ["Surface", "Court", "Round"];
    this.dropCols = [
      "Tournament", "Date", "Best of", "Player_1", "Player_2", "Winner", "Score",
      "Rank_1","Rank_2","Pts_1","Pts_2","Odd_1","Odd_2"
    ];
    this.labelCol = "y";
    this.catLevels = {};
    this.scaler = { mean: {}, std: {} };
    this.featureNames = [];
    this.featureIndexMap = {};
    this.matchRecords = [];
    this.playerNames = [];
    this.matchIndex = new Map();
    this.opponentMap = new Map();
    this.playerStats = new Map();
    this.categoryOptions = new Map();
    this.ageMap = new Map();
    this.ageGroupLevels = AGE_GROUPS.map((g) => g.label);
    this._flipOnReverse = new Set([
      "rank_diff", "pts_diff", "odd_diff", "h2h_advantage", "surface_winrate_adv"
    ]);
    this.playerFeaturePairs = [
      ["age_1", "age_2"],
      ["streak_1", "streak_2"],
      ["streak_value_1", "streak_value_2"],
      ["recent_win_rate_5_1", "recent_win_rate_5_2"],
      ["recent_win_rate_10_1", "recent_win_rate_10_2"],
      ["fatigue_7d_1", "fatigue_7d_2"],
      ["fatigue_14d_1", "fatigue_14d_2"],
      ["fatigue_30d_1", "fatigue_30d_2"],
      ["surface_trend_1", "surface_trend_2"],
    ];
  }

  async loadCSVText(csvText) {
    const delimiter = this._detectDelimiter(csvText);
    const rows = this._parseCSV(csvText, delimiter);
    if (rows.length === 0) throw new Error("Empty CSV file.");
    const headers = rows[0].map(h => (h ?? "").trim());
    const dataRows = rows.slice(1);

    await this._ensureAgesLoaded();

    const raw = dataRows.map((r) => {
      const obj = {};
      headers.forEach((h, i) => (obj[h] = r[i] === undefined ? "" : r[i]));
      return obj;
    });

    if (!headers.includes(this.labelCol)) {
      throw new Error(`Label column "${this.labelCol}" not found in CSV.`);
    }

    const missingNumeric = this.numericCols.filter((c) => !headers.includes(c));
    const missingCategorical = this.categoricalCols.filter((c) => !headers.includes(c));
    if (missingCategorical.length > 0) {
      const missing = [];
      if (missingNumeric.length > 0) missing.push(`numeric: ${missingNumeric.join(", ")}`);
      if (missingCategorical.length > 0) missing.push(`categorical: ${missingCategorical.join(", ")}`);
      throw new Error(`Missing expected columns — ${missing.join("; ")}`);
    }
    if (missingNumeric.length > 0) {
      console.warn("Missing numeric columns in CSV, will impute medians:", missingNumeric.join(", "));
    }

    const metaRows = raw.map((row) => ({
      player1: (row["Player_1"] ?? "").toString().trim(),
      player2: (row["Player_2"] ?? "").toString().trim(),
      date: (row["Date"] ?? "").toString().trim(),
      surface: (row["Surface"] ?? "").toString().trim(),
      court: (row["Court"] ?? "").toString().trim(),
      round: (row["Round"] ?? "").toString().trim(),
      numeric: {},
      categorical: {},
      label: NaN,
      timestamp: this._parseDate((row["Date"] ?? "").toString().trim()),
      rank1: this._toNumber(row["Rank_1"]),
      rank2: this._toNumber(row["Rank_2"]),
      pts1: this._toNumber(row["Pts_1"]),
      pts2: this._toNumber(row["Pts_2"]),
      winner: (row["Winner"] ?? "").toString().trim(),
      score: (row["Score"] ?? "").toString().trim(),
      ageGroups: {},
      playerFeatures: {},
    }));

    // Drop leakage columns; cast types and enrich metadata snapshot
    raw.forEach((row, idx) => {
      const meta = metaRows[idx];
      row[this.labelCol] = this._toNumber(row[this.labelCol]);
      meta.label = row[this.labelCol];
      for (const c of this.numericCols) {
        const num = this._toNumber(row[c]);
        row[c] = this._sanitizeNumeric(c, num);
        meta.numeric[c] = row[c];
      }
      for (const col of this.categoricalCols) {
        const str = (row[col] ?? "").toString().trim();
        row[col] = str;
        meta.categorical[col] = str;
      }
      for (const c of this.dropCols) {
        if (c in row) delete row[c];
      }
    });

    const medians = this._computeMedians(raw);
    const ageFallback = Number.isFinite(medians.age_1)
      ? medians.age_1
      : (Number.isFinite(medians.age_2) ? medians.age_2 : 27);

    raw.forEach((row, idx) => {
      const meta = metaRows[idx];
      for (const c of this.numericCols) {
        if (!Number.isFinite(row[c])) {
          const fallback = Number.isFinite(medians[c]) ? medians[c] : 0;
          row[c] = c.startsWith("age_") ? ageFallback : fallback;
        }
        meta.numeric[c] = row[c];
      }
      this._normalizeSurfaceTrend(row, meta);
      const ageGroup = this._resolveAgeGroups(meta.player1, meta.player2, row, meta.date);
      row.age_group_1 = ageGroup.age_group_1;
      row.age_group_2 = ageGroup.age_group_2;
      meta.ageGroups = ageGroup;
      meta.playerFeatures = this._buildPlayerFeatureView(row, meta);
    });

    const filtered = [];
    const filteredMeta = [];
    raw.forEach((row, idx) => {
      if (!Number.isFinite(row[this.labelCol])) return;
      filtered.push(row);
      filteredMeta.push(metaRows[idx]);
    });
    this._prepareMatchIndex(filteredMeta);
    if (filtered.length < 10) throw new Error(`Too few valid rows: ${filtered.length}`);

    // Stratified split on raw rows to avoid leakage
    const { trainRows, testRows } = this._splitRowsStratified(filtered, 0.2, 42);

    // Fit categorical levels and scalers only on training data
    this._fitCategoricals(trainRows);
    const { X: X_train_raw, y: y_train, featureNames } = this._buildDesignMatrix(trainRows);
    const cleanedFeatureNames = this._cleanFeatureNames(featureNames);
    this.featureNames = cleanedFeatureNames;
    this.featureIndexMap = this.featureNames.reduce((acc, f, i) => { acc[f] = i; return acc; }, {});
    this._fitScaler(X_train_raw, cleanedFeatureNames);

    const X_train_scaled = this._transformWithScaler(X_train_raw, cleanedFeatureNames);
    const { X: X_test_raw, y: y_test } = this._buildDesignMatrix(testRows, cleanedFeatureNames);
    const X_test_scaled = this._transformWithScaler(X_test_raw, cleanedFeatureNames);

    // To tensors
    const xTrainTensor = tf.tensor2d(X_train_scaled, [X_train_scaled.length, featureNames.length], "float32");
    const yTrainTensor = tf.tensor2d(y_train.map(v => [v]), [y_train.length, 1], "float32");
    const xTestTensor = tf.tensor2d(X_test_scaled, [X_test_scaled.length, featureNames.length], "float32");
    const yTestTensor = tf.tensor2d(y_test.map(v => [v]), [y_test.length, 1], "float32");

    return {
      X_train: xTrainTensor, y_train: yTrainTensor,
      X_test: xTestTensor, y_test: yTestTensor,
      featureNames: this.featureNames,
      featureIndexMap: this.featureIndexMap,
      artifacts: {
        catLevels: this.catLevels,
        scaler: this.scaler,
        numericCols: this.numericCols,
        categoricalCols: this.categoricalCols,
        featureNames: this.featureNames,
        featureIndexMap: this.featureIndexMap,
        ageGroupLevels: this.ageGroupLevels,
      }
    };
  }

  getPlayerNames() {
    return this.playerNames.slice();
  }

  getOpponentsFor(player) {
    if (!player) return [];
    const opponents = this.opponentMap.get(player);
    return opponents ? opponents.slice() : [];
  }

  getPlayerSnapshot(player) {
    if (!player) return null;
    return this.playerStats.get(player) || null;
  }

  getCategoryOptions(column) {
    const opts = this.categoryOptions.get(column);
    return opts ? opts.slice() : [];
  }

  getLatestAutoFeatures(player1, player2) {
    if (!player1 || !player2 || player1 === player2) return null;
    const forwardKey = this._pairKey(player1, player2);
    let match = this.matchIndex.get(forwardKey)?.[0];
    let forward = true;
    if (!match) {
      const reverseMatches = this.matchIndex.get(this._pairKey(player2, player1));
      if (!reverseMatches || reverseMatches.length === 0) return null;
      match = reverseMatches[0];
      forward = false;
    }
    return this._orientMatchForPlayers(match, player1, player2, forward);
  }

  vectorizeForPredict(userInput) {
    const rowObj = {};
    const age1 = this._sanitizeNumeric("age_1", this._toNumber(userInput.age_1));
    const age2 = this._sanitizeNumeric("age_2", this._toNumber(userInput.age_2));
    const ageGroup = this._computeAgeGroups(age1, age2);
    for (const c of this.numericCols) {
      const v = this._sanitizeNumeric(c, this._toNumber(userInput[c]));
      if (!Number.isFinite(v)) throw new Error(`Numeric input "${c}" missing or invalid.`);
      rowObj[c] = v;
    }
    rowObj.age_1 = Number.isFinite(age1) ? age1 : 0;
    rowObj.age_2 = Number.isFinite(age2) ? age2 : 0;
    rowObj.age_group_1 = ageGroup.age_group_1;
    rowObj.age_group_2 = ageGroup.age_group_2;
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      const provided = (userInput[col] ?? "").toString();
      for (const lvl of levels) {
        const key = `${col}__${lvl}`;
        rowObj[key] = provided === lvl ? 1 : 0;
      }
    }
    for (const level of this.ageGroupLevels) {
      rowObj[`age_group_1__${level}`] = rowObj.age_group_1 === level ? 1 : 0;
      rowObj[`age_group_2__${level}`] = rowObj.age_group_2 === level ? 1 : 0;
    }
    const vec = this.featureNames.map((f) => {
      let v = rowObj[f] ?? 0;
      if (this.numericCols.includes(f)) {
        const mean = this.scaler.mean[f] ?? 0;
        const std = this.scaler.std[f] ?? 1;
        v = std === 0 ? 0 : (v - mean) / std;
      } else if (this._isAgeGroupFeature(f)) {
        v = rowObj[f] ?? 0;
      }
      return v;
    });
    return Float32Array.from(vec);
  }

  _detectDelimiter(text) {
    const firstLine = text.split(/\r?\n/)[0] || "";
    const comma = (firstLine.match(/,/g) || []).length;
    const semicolon = (firstLine.match(/;/g) || []).length;
    return semicolon > comma ? ";" : ",";
  }

  _parseCSV(text, delimiter) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    const rows = [];
    for (const line of lines) {
      const row = [];
      let current = "", inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
          else inQuotes = !inQuotes;
        } else if (ch === delimiter && !inQuotes) {
          row.push(current); current = "";
        } else current += ch;
      }
      row.push(current);
      rows.push(row);
    }
    return rows;
  }

  _toNumber(v) {
    if (v === undefined || v === null || v === "") return NaN;
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }

  _fitCategoricals(rows) {
    for (const col of this.categoricalCols) {
      const set = new Set();
      for (const r of rows) {
        const val = (r[col] ?? "").toString().trim();
        if (val.length > 0) set.add(val);
      }
      this.catLevels[col] = Array.from(set.values()).sort();
    }
  }

  _buildDesignMatrix(rows, featureNames = null) {
    const resolvedFeatureNames = featureNames ? featureNames.slice() : this._featureNamesFromArtifacts();
    const X = [], y = [];
    for (const r of rows) {
      const rowArr = [];
      for (const feat of resolvedFeatureNames) {
        if (this.numericCols.includes(feat)) {
          rowArr.push(r[feat]);
        } else if (this._isAgeGroupFeature(feat)) {
          const { base, level } = this._parseAgeGroupFeature(feat);
          const val = (r[base] ?? "").toString();
          rowArr.push(val === level ? 1 : 0);
        } else {
          const [col, lvl] = feat.split("__");
          const val = (r[col] ?? "").toString();
          rowArr.push(val === lvl ? 1 : 0);
        }
      }
      X.push(rowArr);
      y.push(Math.round(r[this.labelCol]));
    }
    return { X, y, featureNames: resolvedFeatureNames };
  }

  _parseDate(str) {
    const t = Date.parse(str);
    return Number.isFinite(t) ? t : NaN;
  }

  _pairKey(a, b) {
    return `${a}|||${b}`;
  }

  _prepareMatchIndex(metaRows) {
    this.matchRecords = metaRows;
    const players = new Set();
    const index = new Map();
    const opponents = new Map();
    const registerOpponent = (base, opp) => {
      if (!base || !opp) return;
      if (!opponents.has(base)) opponents.set(base, new Set());
      opponents.get(base).add(opp);
    };

    for (const meta of metaRows) {
      if (meta.player1) players.add(meta.player1);
      if (meta.player2) players.add(meta.player2);
      const key = this._pairKey(meta.player1, meta.player2);
      if (!index.has(key)) index.set(key, []);
      index.get(key).push(meta);
      registerOpponent(meta.player1, meta.player2);
      registerOpponent(meta.player2, meta.player1);
    }

    for (const arr of index.values()) {
      arr.sort((a, b) => {
        const ta = Number.isFinite(a.timestamp) ? a.timestamp : -Infinity;
        const tb = Number.isFinite(b.timestamp) ? b.timestamp : -Infinity;
        return tb - ta;
      });
    }
    this.matchIndex = index;
    this.playerNames = Array.from(players.values()).sort((a, b) => a.localeCompare(b));
    this.playerStats = this._buildPlayerStats(metaRows);
    this.categoryOptions = this._collectCategoryOptions(metaRows);
    const sortedOpponents = new Map();
    for (const [player, set] of opponents.entries()) {
      sortedOpponents.set(
        player,
        Array.from(set.values()).sort((a, b) => a.localeCompare(b))
      );
    }
    this.opponentMap = sortedOpponents;
  }

  _orientMatchForPlayers(match, player1, player2, alreadyForward) {
    if (!match) return null;
    const numeric = {};
    const vectorInput = {};
    const player1Snapshot = this.getPlayerSnapshot(player1);
    const player2Snapshot = this.getPlayerSnapshot(player2);

    const orientedNumeric = this._orientNumeric(match.numeric, alreadyForward);
    const ageResolved = this._resolveAgeForScenario(player1, player2, orientedNumeric, match.date);
    Object.assign(numeric, orientedNumeric, ageResolved.numeric);
    Object.assign(vectorInput, orientedNumeric, ageResolved.numeric);

    const categorical = {};
    for (const col of this.categoricalCols) {
      const val = match.categorical[col] ?? "";
      categorical[col] = val;
      vectorInput[col] = val;
    }

    const ageGroups = this._computeAgeGroups(numeric.age_1, numeric.age_2);
    vectorInput.age_1 = numeric.age_1;
    vectorInput.age_2 = numeric.age_2;
    vectorInput.age_group_1 = ageGroups.age_group_1;
    vectorInput.age_group_2 = ageGroups.age_group_2;

    const playerFeatures = this._buildPlayerFeatureView(numeric, {
      player1,
      player2,
      date: match.date,
      ageGroups
    });

    return {
      players: { player1, player2 },
      datasetMatch: {
        player1: match.player1,
        player2: match.player2,
        date: match.date,
        winner: match.winner,
        score: match.score,
        orientation: alreadyForward ? "forward" : "reverse"
      },
      playerSnapshots: {
        [player1]: player1Snapshot || null,
        [player2]: player2Snapshot || null
      },
      numeric,
      categorical,
      ageGroups,
      playerFeatures,
      vectorInput
    };
  }

  _featureNamesFromArtifacts() {
    const featureNames = [...this.numericCols];
    for (const level of this.ageGroupLevels) {
      featureNames.push(`age_group_1__${level}`);
      featureNames.push(`age_group_2__${level}`);
    }
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      for (const lvl of levels) featureNames.push(`${col}__${lvl}`);
    }
    return this._cleanFeatureNames(featureNames);
  }

  _cleanFeatureNames(featureNames) {
    const incoming = Array.isArray(featureNames) ? featureNames : [];
    const cleaned = incoming.filter((name) => name && name !== this.labelCol);
    if (incoming.length !== cleaned.length) {
      console.warn(`Label column "${this.labelCol}" was present in features and has been removed.`);
    }
    return cleaned;
  }

  _orientNumeric(numeric, alreadyForward) {
    const oriented = {};
    const swap = (a, b) => ({ a: alreadyForward ? a : b, b: alreadyForward ? b : a });
    const rankDiff = alreadyForward ? numeric.rank_diff : -numeric.rank_diff;
    oriented.rank_diff = this._sanitizeNumeric("rank_diff", rankDiff);
    oriented.pts_diff = this._sanitizeNumeric("pts_diff", alreadyForward ? numeric.pts_diff : -numeric.pts_diff);
    oriented.odd_diff = this._sanitizeNumeric("odd_diff", alreadyForward ? numeric.odd_diff : -numeric.odd_diff);
    oriented.h2h_advantage = this._sanitizeNumeric("h2h_advantage", alreadyForward ? numeric.h2h_advantage : -numeric.h2h_advantage);
    oriented.last_winner = this._sanitizeNumeric("last_winner", alreadyForward ? numeric.last_winner : (1 - numeric.last_winner));
    oriented.surface_winrate_adv = this._sanitizeNumeric("surface_winrate_adv", alreadyForward ? numeric.surface_winrate_adv : -numeric.surface_winrate_adv);

    for (const [a, b] of this.playerFeaturePairs) {
      const { a: v1, b: v2 } = swap(numeric[a], numeric[b]);
      oriented[a] = this._sanitizeNumeric(a, v1);
      oriented[b] = this._sanitizeNumeric(b, v2);
    }
    return oriented;
  }

  _resolveAgeForScenario(player1, player2, numeric, dateStr) {
    const dateTs = this._parseDate(dateStr);
    const refDate = new Date(`${SCENARIO_YEAR}-01-01T00:00:00Z`).getTime();
    const resolved = { numeric: {} };
    const age1 = this._resolveAge(player1, numeric.age_1, refDate, dateTs);
    const age2 = this._resolveAge(player2, numeric.age_2, refDate, dateTs);
    resolved.numeric.age_1 = age1;
    resolved.numeric.age_2 = age2;
    return resolved;
  }

  _computeAgeGroups(age1, age2) {
    const resolve = (age) => {
      const val = Number.isFinite(age) ? age : 0;
      for (const g of AGE_GROUPS) {
        if (val >= g.min && val < g.max) return g.label;
      }
      return AGE_GROUPS[AGE_GROUPS.length - 1].label;
    };
    return { age_group_1: resolve(age1), age_group_2: resolve(age2) };
  }

  _sanitizeNumeric(key, value) {
    let v = Number.isFinite(value) ? value : NaN;
    if (key.startsWith("surface_trend")) {
      if (!Number.isFinite(v)) v = 0;
      v = Math.max(-1, Math.min(1, v));
    }
    return v;
  }

  _computeMedians(rows) {
    const medians = {};
    for (const c of this.numericCols) {
      const values = rows.map((r) => this._sanitizeNumeric(c, r[c])).filter((v) => Number.isFinite(v));
      medians[c] = this._median(values);
    }
    return medians;
  }

  _median(arr) {
    if (!arr || arr.length === 0) return NaN;
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
    return sorted[mid];
  }

  _normalizeSurfaceTrend(row, meta) {
    const clamp = (val) => {
      if (!Number.isFinite(val)) return 0;
      return Math.max(-1, Math.min(1, val));
    };
    row.surface_trend_1 = clamp(row.surface_trend_1);
    row.surface_trend_2 = clamp(row.surface_trend_2);
    meta.numeric.surface_trend_1 = row.surface_trend_1;
    meta.numeric.surface_trend_2 = row.surface_trend_2;
  }

  _buildPlayerFeatureView(sourceNumeric, ctx) {
    const ageGroups = ctx?.ageGroups || this._computeAgeGroups(sourceNumeric.age_1, sourceNumeric.age_2);
    return {
      [ctx.player1]: {
        age: sourceNumeric.age_1,
        ageGroup: ageGroups.age_group_1,
        streak: sourceNumeric.streak_1,
        streakValue: sourceNumeric.streak_value_1,
        recent5: sourceNumeric.recent_win_rate_5_1,
        recent10: sourceNumeric.recent_win_rate_10_1,
        fatigue7: sourceNumeric.fatigue_7d_1,
        fatigue14: sourceNumeric.fatigue_14d_1,
        fatigue30: sourceNumeric.fatigue_30d_1,
        surfaceTrend: sourceNumeric.surface_trend_1,
        date: ctx.date,
      },
      [ctx.player2]: {
        age: sourceNumeric.age_2,
        ageGroup: ageGroups.age_group_2,
        streak: sourceNumeric.streak_2,
        streakValue: sourceNumeric.streak_value_2,
        recent5: sourceNumeric.recent_win_rate_5_2,
        recent10: sourceNumeric.recent_win_rate_10_2,
        fatigue7: sourceNumeric.fatigue_7d_2,
        fatigue14: sourceNumeric.fatigue_14d_2,
        fatigue30: sourceNumeric.fatigue_30d_2,
        surfaceTrend: sourceNumeric.surface_trend_2,
        date: ctx.date,
      }
    };
  }

  async _ensureAgesLoaded() {
    if (this.ageMap && this.ageMap.size > 0) return;
    try {
      const data = await this._loadAgeCSV();
      if (data && data.size > 0) this.ageMap = data;
    } catch (err) {
      console.warn("Age CSV not loaded; continuing with dataset values only", err);
    }
  }

  async _loadAgeCSV() {
    const res = await fetch(`./wta_age.csv?v=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return new Map();
    const text = await res.text();
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    const map = new Map();
    for (const line of lines.slice(1)) {
      const [name, birth] = line.split(",").map((v) => v.trim());
      if (!name || !birth) continue;
      map.set(name, birth);
    }
    return map;
  }

  _resolveAge(player, existingAge, refTs, fallbackTs) {
    if (!player) return Number.isFinite(existingAge) ? existingAge : NaN;
    const birthStr = this.ageMap.get(player);
    if (!birthStr) return Number.isFinite(existingAge) ? existingAge : NaN;
    const birthTs = this._parseDate(birthStr);
    const reference = Number.isFinite(refTs) ? refTs : fallbackTs;
    if (!Number.isFinite(birthTs) || !Number.isFinite(reference)) {
      return Number.isFinite(existingAge) ? existingAge : NaN;
    }
    const years = (reference - birthTs) / (365.25 * 24 * 3600 * 1000);
    return Number.isFinite(years) ? Math.max(14, years) : existingAge;
  }

  _resolveAgeGroups(player1, player2, row, matchDate) {
    const age1 = this._resolveAge(player1, row.age_1, this._parseDate(matchDate));
    const age2 = this._resolveAge(player2, row.age_2, this._parseDate(matchDate));
    row.age_1 = Number.isFinite(age1) ? age1 : row.age_1;
    row.age_2 = Number.isFinite(age2) ? age2 : row.age_2;
    return this._computeAgeGroups(row.age_1, row.age_2);
  }

  _isAgeGroupFeature(name) {
    return name.startsWith("age_group_1__") || name.startsWith("age_group_2__");
  }

  _parseAgeGroupFeature(name) {
    const [base, level] = name.split("__");
    return { base, level };
  }

  _buildPlayerStats(metaRows) {
    const stats = new Map();
    const update = (player, rank, pts, timestamp, dateStr) => {
      if (!player) return;
      const time = Number.isFinite(timestamp) ? timestamp : -Infinity;
      const existing = stats.get(player);
      if (!existing || time > existing.timestamp) {
        stats.set(player, {
          rank: Number.isFinite(rank) ? rank : null,
          pts: Number.isFinite(pts) ? pts : null,
          timestamp: time,
          date: dateStr || null
        });
      }
    };
    for (const meta of metaRows) {
      update(meta.player1, meta.rank1, meta.pts1, meta.timestamp, meta.date);
      update(meta.player2, meta.rank2, meta.pts2, meta.timestamp, meta.date);
    }
    return stats;
  }

  _collectCategoryOptions(metaRows) {
    const sets = new Map();
    for (const col of this.categoricalCols) {
      sets.set(col, new Set());
    }
    for (const meta of metaRows) {
      for (const col of this.categoricalCols) {
        const val = meta.categorical[col];
        if (val && sets.has(col)) {
          sets.get(col).add(val);
        }
      }
    }
    const map = new Map();
    for (const [col, set] of sets.entries()) {
      map.set(col, Array.from(set.values()).sort((a, b) => a.localeCompare(b)));
    }
    return map;
  }

  _fitScaler(X, featureNames) {
    for (const c of this.numericCols) {
      const idx = featureNames.indexOf(c);
      if (idx === -1) continue;
      const n = X.length;
      let sum = 0, sumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = X[i][idx];
        sum += v; sumSq += v * v;
      }
      const mean = sum / Math.max(1, n);
      const variance = Math.max(0, sumSq / Math.max(1, n) - mean * mean);
      const std = Math.sqrt(variance);
      this.scaler.mean[c] = mean;
      this.scaler.std[c] = std;
    }
  }

  _transformWithScaler(X, featureNames) {
    const X2 = X.map((row) => row.slice());
    for (const c of this.numericCols) {
      const idx = featureNames.indexOf(c);
      if (idx === -1) continue;
      const mean = this.scaler.mean[c] ?? 0;
      const std = this.scaler.std[c] ?? 1;
      for (let i = 0; i < X2.length; i++) {
        const v = X2[i][idx];
        X2[i][idx] = std === 0 ? 0 : (v - mean) / std;
      }
    }
    return X2;
  }

  _splitRowsStratified(rows, testSize = 0.2, seed = 42) {
    const idxByClass = new Map();
    for (let i = 0; i < rows.length; i++) {
      const label = Math.round(rows[i][this.labelCol]);
      if (!idxByClass.has(label)) idxByClass.set(label, []);
      idxByClass.get(label).push(i);
    }

    const rng = this._mulberry32(seed);
    const shuffle = (arr) => {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    };

    const total = rows.length;
    const targetTest = Math.max(1, Math.round(total * testSize));
    const testIdx = new Set();

    let assigned = 0;
    const classes = Array.from(idxByClass.keys());
    for (const cls of classes) {
      const arr = shuffle(idxByClass.get(cls));
      const clsTotal = arr.length;
      if (clsTotal === 0) continue;
      const desired = Math.max(clsTotal > 0 ? 1 : 0, Math.round(clsTotal * testSize));
      const take = Math.min(clsTotal, desired);
      for (let i = 0; i < take; i++) {
        if (assigned >= targetTest) break;
        testIdx.add(arr[i]);
        assigned++;
      }
    }

    // If still short on test samples (e.g., due to rounding), fill from remaining
    if (assigned < targetTest) {
      const allIdx = Array.from({ length: total }, (_, i) => i);
      shuffle(allIdx);
      for (const idx of allIdx) {
        if (assigned >= targetTest) break;
        if (!testIdx.has(idx)) {
          testIdx.add(idx);
          assigned++;
        }
      }
    }

    const trainRows = [];
    const testRows = [];
    for (let i = 0; i < rows.length; i++) {
      if (testIdx.has(i)) testRows.push(rows[i]);
      else trainRows.push(rows[i]);
    }
    return { trainRows, testRows };
  }

  _mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
}
