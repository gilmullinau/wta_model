// data-loader.js
// Reads CSV text, preprocesses: drop leakage columns, numeric scaling, one-hot for categoricals,
// stratified split train/test, returns tf.Tensors and feature metadata.
import { augmentWithMirrorExamples, buildSequences } from "./sequence-builder.js";

const tf = window.tf;
const SCENARIO_YEAR = 2025;

const BASE_GRU_SEQUENCE_FEATURES = [
  // Competitive diffs / matchup context (per player orientation)
  "rank_diff",
  "pts_diff",
  "odd_diff",
  "h2h_advantage",
  "last_winner",
  "surface_winrate_adv",
  "year",

  // Dynamic form / fatigue
  "recent_win_rate_5",
  "recent_win_rate_10",
  "streak_value",
  "fatigue_7d",
  "fatigue_14d",
  "fatigue_30d",

  // Surface momentum
  "surface_trend",
];

// Accuracy-focused full feature list (14 engineered columns).
export const GRU_SEQUENCE_FEATURES = BASE_GRU_SEQUENCE_FEATURES.slice();

const DEPRECATED_SEQUENCE_FEATURES = new Set([
  "rolling_win_rate_10",
  "streak",
]);

const REQUIRED_SEQUENCE_COLUMNS = [
  "y",
  "match_date",
  "Date",
  "Player_1",
  "Player_2",
  "Rank_1",
  "Rank_2",
  "Pts_1",
  "Pts_2",
  "Odd_1",
  "Odd_2",
  "Tournament",
  ...GRU_SEQUENCE_FEATURES,
  "Surface",
  "Court",
  "Round",
];

export class DataLoader {
  constructor(modelType = "MLP", seqLen = 10, options = {}) {
    this.modelType = modelType;
    this.seqLen = seqLen;
    this.enableSequenceDebug = Boolean(options.enableSequenceDebug);
    this.featureListMLP = [
      "rank_diff", "pts_diff", "odd_diff",
      "h2h_advantage", "last_winner", "surface_winrate_adv", "year"
    ];
    this.featureListGRU = GRU_SEQUENCE_FEATURES.slice();
    this.numericCols = this.featureListMLP.slice();
    this.sequenceFeatureCols = this._buildSequenceFeatureList();
    this.categoricalCols = ["Surface", "Court", "Round"];
    this.dropCols = [
      "Tournament", "Date", "Best of", "Best_of", "Player_1", "Player_2", "Winner", "Score",
      "Rank_1","Rank_2","Pts_1","Pts_2","Odd_1","Odd_2",
      // Explicitly drop legacy surface win-rate columns so they cannot leak back in exported CSVs
      "surface_win_rate_hard_5", "surface_win_rate_clay_5", "surface_win_rate_grass_5",
    ];
    this.labelCol = "y";
    this.catLevels = {};
    this.scaler = { mean: {}, std: {} };
    this.featureNames = [];
    this.matchRecords = [];
    this.playerNames = [];
    this.matchIndex = new Map();
    this.opponentMap = new Map();
    this.playerStats = new Map();
    this.categoryOptions = new Map();
    this.sequenceRows = [];
    this.cleanedRows = [];
    this.meta = {
      modelType,
      featureList: [],
      featureIndexMap: {},
      seqLen: this.isSequenceMode() ? seqLen : null,
      mean: {},
      std: {},
    };
    this._flipOnReverse = new Set([
      "rank_diff", "pts_diff", "odd_diff", "h2h_advantage", "surface_winrate_adv", "last_winner"
    ]);
    this.sequenceDebug = null;
  }

  isSequenceMode() {
    return this.modelType === "GRU" || this.modelType === "RNN";
  }

  async loadCSVText(csvText) {
    const delimiter = this._detectDelimiter(csvText);
    const rows = this._parseCSV(csvText, delimiter);
    if (rows.length === 0) throw new Error("Empty CSV file.");
    const headers = rows[0].map(h => (h ?? "").trim());
    const dataRows = rows.slice(1);

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
    if (missingNumeric.length > 0 || missingCategorical.length > 0) {
      const missing = [];
      if (missingNumeric.length > 0) missing.push(`numeric: ${missingNumeric.join(", ")}`);
      if (missingCategorical.length > 0) missing.push(`categorical: ${missingCategorical.join(", ")}`);
      throw new Error(`Missing expected columns — ${missing.join("; ")}`);
    }

    if (this.isSequenceMode()) {
      const missingReq = REQUIRED_SEQUENCE_COLUMNS.filter((c) => !headers.includes(c));
      if (missingReq.length > 0) {
        throw new Error(`Missing required CSV columns for sequences: ${missingReq.join(", ")}`);
      }
    }

    this.sequenceFeatureCols = this._buildSequenceFeatureList(headers);

    const metaRows = raw.map((row) => ({
      player1: (row["Player_1"] ?? "").toString().trim(),
      player2: (row["Player_2"] ?? "").toString().trim(),
      date: (row["match_date"] ?? row["Date"] ?? "").toString().trim(),
      surface: (row["Surface"] ?? "").toString().trim(),
      court: (row["Court"] ?? "").toString().trim(),
      round: (row["Round"] ?? "").toString().trim(),
      numeric: {},
      categorical: {},
      label: NaN,
      timestamp: this._parseDate((row["match_date"] ?? row["Date"] ?? "").toString().trim()),
      rank1: this._toNumber(row["Rank_1"]),
      rank2: this._toNumber(row["Rank_2"]),
      pts1: this._toNumber(row["Pts_1"]),
      pts2: this._toNumber(row["Pts_2"]),
      winner: (row["Winner"] ?? "").toString().trim(),
      score: (row["Score"] ?? "").toString().trim()
    }));

    // Drop leakage columns; cast types and enrich metadata snapshot
    raw.forEach((row, idx) => {
      const meta = metaRows[idx];
      row[this.labelCol] = this._toNumber(row[this.labelCol]);
      meta.label = row[this.labelCol];
      for (const c of this.numericCols) {
        const num = this._toNumber(row[c]);
        row[c] = num;
        meta.numeric[c] = num;
      }
      for (const c of this.sequenceFeatureCols) {
        const num = this._toNumber(row[c]);
        row[c] = Number.isFinite(num) ? num : 0;
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

    this._recomputeSurfaceTrend(raw, metaRows);

    const filtered = [];
    const filteredMeta = [];
    raw.forEach((row, idx) => {
      if (!Number.isFinite(row[this.labelCol])) return;
      for (const c of this.numericCols) {
        if (!Number.isFinite(row[c])) return;
      }
      if (this.isSequenceMode()) {
        for (const feat of this.sequenceFeatureCols) {
          const val = this._toNumber(row[feat]);
          if (!Number.isFinite(val)) return;
        }
      }
      const meta = metaRows[idx];
      const augmented = {
        ...row,
        player: meta.player1,
        player1: meta.player1,
        player2: meta.player2,
        date: meta.date,
        timestamp: meta.timestamp,
        __meta: meta,
      };
      filtered.push(augmented);
      filteredMeta.push(meta);
    });
    this.cleanedRows = filtered.map((row) => {
      const copy = { ...row };
      delete copy.__meta;
      return copy;
    });
    this.sequenceRows = filtered.map((row) => ({ ...row }));
    this.sequenceRowsRaw = filtered.map((row) => ({ ...row }));
    this._prepareMatchIndex(filteredMeta);
    if (filtered.length < 10) throw new Error(`Too few valid rows: ${filtered.length}`);

    // Stratified split on raw rows to avoid leakage
    const { trainRows, testRows } = this._splitRowsStratified(filtered, 0.2, 42);

    // Fit categorical levels and scalers only on training data
    if (this.isSequenceMode()) {
      const featureList = this.sequenceFeatureCols.slice();
      this._validateGruFeatures(headers, featureList);
      const { mean, std } = this._computeScalerForRows(trainRows, featureList);

      const seqTrain = buildSequences(trainRows, this.seqLen, featureList, {
        mean,
        std,
        flipOnReverse: this._flipOnReverse,
      });
      const seqTest = buildSequences(testRows, this.seqLen, featureList, {
        mean,
        std,
        flipOnReverse: this._flipOnReverse,
      });

      const trainAug = augmentWithMirrorExamples(seqTrain.X, seqTrain.y, seqTrain.meta.sampleInfo);
      const testAug = augmentWithMirrorExamples(seqTest.X, seqTest.y, seqTest.meta.sampleInfo);

      const seqTrainAug = {
        ...seqTrain,
        X: trainAug.X,
        y: trainAug.y,
        meta: { ...seqTrain.meta, sampleInfo: trainAug.sampleInfo || seqTrain.meta.sampleInfo },
        stats: { ...seqTrain.stats, numSamples: trainAug.X.length },
      };

      const seqTestAug = {
        ...seqTest,
        X: testAug.X,
        y: testAug.y,
        meta: { ...seqTest.meta, sampleInfo: testAug.sampleInfo || seqTest.meta.sampleInfo },
        stats: { ...seqTest.stats, numSamples: testAug.X.length },
      };

      if (seqTrain.stats.paddingPercent > 90) {
        throw new Error("Sequence length exceeds available match history for most players");
      }

      const numFeatures = seqTrain.meta.numFeatures;
      const X_train = tf.tensor3d(seqTrainAug.X, [seqTrainAug.stats.numSamples, this.seqLen, numFeatures], "float32");
      const y_train = tf.tensor1d(seqTrainAug.y, "float32");
      const X_test = tf.tensor3d(seqTestAug.X, [seqTestAug.stats.numSamples, this.seqLen, numFeatures], "float32");
      const y_test = tf.tensor1d(seqTestAug.y, "float32");

      console.log("GRU tensors created", X_train instanceof tf.Tensor, y_train instanceof tf.Tensor, X_test instanceof tf.Tensor, y_test instanceof tf.Tensor);
      this.X_train = X_train;
      this.y_train = y_train;
      this.X_test = X_test;
      this.y_test = y_test;

      const combinedFeatureList = seqTrain.meta.featureList.slice();
      this.featureNames = combinedFeatureList.slice();
      this.sequenceDebug = this.enableSequenceDebug
        ? buildSequences(filtered, this.seqLen, featureList, { mean, std, flipOnReverse: this._flipOnReverse })
        : seqTrainAug;
      this.meta = {
        modelType: this.modelType,
        featureList: combinedFeatureList,
        featureIndexMap: seqTrain.meta.featureIndexMap,
        seqLen: this.seqLen,
        featureCount: combinedFeatureList.length,
        mean,
        std,
        stats: seqTrainAug.stats,
        baseFeatureList: featureList.slice(),
      };

      return {
        X_train,
        y_train,
        X_test,
        y_test,
        featureNames: this.featureNames,
        artifacts: {
          scaler: { mean, std },
          featureNames: this.featureNames,
          featureCount: combinedFeatureList.length,
          featureIndexMap: seqTrain.meta.featureIndexMap,
          modelType: this.modelType,
          seqLen: this.seqLen,
          stats: seqTrainAug.stats,
          baseFeatureList: featureList.slice(),
        }
      };
    }

    this._fitCategoricals(trainRows);
    const { X: X_train_raw, y: y_train, featureNames } = this._buildDesignMatrix(trainRows);
    this.featureNames = featureNames;
    this._fitScaler(X_train_raw, featureNames);

    const X_train_scaled = this._transformWithScaler(X_train_raw, featureNames);
    const { X: X_test_raw, y: y_test } = this._buildDesignMatrix(testRows, featureNames);
    const X_test_scaled = this._transformWithScaler(X_test_raw, featureNames);

    // To tensors
    const xTrainTensor = tf.tensor2d(X_train_scaled, [X_train_scaled.length, featureNames.length], "float32");
    const yTrainTensor = tf.tensor2d(y_train.map(v => [v]), [y_train.length, 1], "float32");
    const xTestTensor = tf.tensor2d(X_test_scaled, [X_test_scaled.length, featureNames.length], "float32");
    const yTestTensor = tf.tensor2d(y_test.map(v => [v]), [y_test.length, 1], "float32");

    this.meta = {
      modelType: "MLP",
      featureList: this.featureNames.slice(),
      seqLen: null,
      mean: this.scaler.mean,
      std: this.scaler.std,
    };

    return {
      X_train: xTrainTensor, y_train: yTrainTensor,
      X_test: xTestTensor, y_test: yTestTensor,
      featureNames: this.featureNames,
      artifacts: {
        catLevels: this.catLevels,
        scaler: this.scaler,
        numericCols: this.numericCols,
        categoricalCols: this.categoricalCols,
        featureNames: this.featureNames,
        modelType: "MLP",
        seqLen: null,
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

  getSequenceFeatureList() {
    return this.sequenceFeatureCols.slice();
  }

  getSequenceRows() {
    return this.sequenceRows.slice();
  }

  getSequenceDebug() {
    return this.sequenceDebug ? { ...this.sequenceDebug, X: this.sequenceDebug.X.slice(), y: this.sequenceDebug.y.slice() } : null;
  }

  getCleanedRows() {
    return this.cleanedRows.map((row) => ({ ...row }));
  }

  buildSequenceInputForMatch(player1, player2, seqLen = this.seqLen) {
    if (!this.isSequenceMode()) {
      throw new Error("Sequence mode (GRU/RNN) is required to build inputs.");
    }
    if (!player1 || !player2) throw new Error("Both Player 1 and Player 2 are required for prediction.");
    const baseFeatures = this.meta.baseFeatureList?.length ? this.meta.baseFeatureList : this.sequenceFeatureCols;
    const featureList = baseFeatures.flatMap((f) => [`p1_${f}`, `p2_${f}`]);
    const mean = this.meta.mean || {};
    const std = this.meta.std || {};

    const history1 = this._collectPlayerHistory(player1, baseFeatures, mean, std);
    const history2 = this._collectPlayerHistory(player2, baseFeatures, mean, std, true);

    const seq1 = this._padSequence(history1.slice(-seqLen), seqLen, baseFeatures.length);
    const seq2 = this._padSequence(history2.slice(-seqLen), seqLen, baseFeatures.length);
    const sequence = seq1.map((step, idx) => step.concat(seq2[idx]));

    const tensor = tf.tensor3d([sequence], [1, seqLen, baseFeatures.length * 2], "float32");
    const latest1 = history1.length ? history1[history1.length - 1].date : "";
    const latest2 = history2.length ? history2[history2.length - 1].date : "";
    return {
      tensor,
      sequence,
      featureList,
      meta: {
        player1,
        player2,
        latestDate: latest1 || latest2 || "",
        padded: history1.length < seqLen || history2.length < seqLen,
        usedRows: Math.min(seqLen, Math.min(history1.length, history2.length)),
      }
    };
  }

  buildPredictSequence(player, seqLen = this.seqLen) {
    if (!player) throw new Error("Player name is required for GRU prediction.");
    const featureList = this.sequenceFeatureCols.slice();
    const rows = this.sequenceRows
      .filter((r) => (r.player || "").toString() === player)
      .sort((a, b) => {
        const ta = Number.isFinite(a.timestamp) ? a.timestamp : -Infinity;
        const tb = Number.isFinite(b.timestamp) ? b.timestamp : -Infinity;
        return ta - tb;
      });

    const sequence = [];
    const padding = Math.max(0, seqLen - rows.length);
    for (let i = 0; i < padding; i++) {
      sequence.push(Array.from({ length: featureList.length }, () => 0));
    }

    const recent = rows.slice(-seqLen);
    for (const r of recent) {
      const step = featureList.map((f) => {
        const val = this._toNumber(r[f]);
        return Number.isFinite(val) ? val : 0;
      });
      sequence.push(step);
    }

    return {
      sequence,
      featureList,
      meta: {
        player,
        latestDate: recent.length > 0 ? (recent[recent.length - 1].date || "") : "",
        padded: padding > 0,
        usedRows: recent.length,
      }
    };
  }

  _recomputeSurfaceTrend(rows, metaRows) {
    const history = new Map();
    const zipped = rows.map((row, idx) => ({
      row,
      meta: metaRows[idx],
      timestamp: Number.isFinite(metaRows[idx]?.timestamp) ? metaRows[idx].timestamp : -Infinity,
    }));

    zipped.sort((a, b) => a.timestamp - b.timestamp);

    for (const { row, meta } of zipped) {
      const player = (meta?.player1 || meta?.player || row.player || "").toString().trim();
      const surface = (meta?.surface || row.Surface || row.surface || "").toString().trim();
      if (!player || !surface) {
        row.surface_trend = 0;
        if (meta?.numeric) meta.numeric.surface_trend = 0;
        continue;
      }

      const key = `${player}|||${surface}`;
      const past = history.get(key) || [];

      const short = past.slice(-5);
      const long = past.slice(-15);
      const shortWr = short.length ? short.reduce((a, b) => a + b, 0) / short.length : 0;
      const longWr = long.length ? long.reduce((a, b) => a + b, 0) / long.length : 0;
      const trend = Math.max(-1, Math.min(1, shortWr - longWr));

      const cleanTrend = Number.isFinite(trend) ? trend : 0;
      row.surface_trend = cleanTrend;
      if (meta?.numeric) meta.numeric.surface_trend = cleanTrend;

      const isWinRaw = this._toNumber(row[this.labelCol]);
      const isWin = Number.isFinite(isWinRaw) && isWinRaw >= 0.5 ? 1 : 0;
      const updated = past.concat(isWin);
      history.set(key, updated.length > 15 ? updated.slice(-15) : updated);
    }
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
    for (const c of this.numericCols) {
      const v = this._toNumber(userInput[c]);
      if (!Number.isFinite(v)) throw new Error(`Numeric input "${c}" missing or invalid.`);
      rowObj[c] = v;
    }
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      const provided = (userInput[col] ?? "").toString();
      for (const lvl of levels) {
        const key = `${col}__${lvl}`;
        rowObj[key] = provided === lvl ? 1 : 0;
      }
    }
    const vec = this.featureNames.map((f) => {
      let v = rowObj[f] ?? 0;
      if (this.numericCols.includes(f)) {
        const mean = this.scaler.mean[f] ?? 0;
        const std = this.scaler.std[f] ?? 1;
        v = std === 0 ? 0 : (v - mean) / std;
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

  _buildSequenceFeatureList(headers) {
    const base = GRU_SEQUENCE_FEATURES.slice();
    if (!headers || headers.length === 0) return base.filter((f) => !DEPRECATED_SEQUENCE_FEATURES.has(f));
    const missing = base.filter((f) => !headers.includes(f) && !DEPRECATED_SEQUENCE_FEATURES.has(f));
    if (missing.length > 0) {
      throw new Error(`Missing feature(s) for sequence model: ${missing.join(", ")}`);
    }
    return base.filter((f) => !DEPRECATED_SEQUENCE_FEATURES.has(f));
  }

  _buildDesignMatrix(rows, featureNames = null) {
    const resolvedFeatureNames = featureNames ? featureNames.slice() : this._featureNamesFromArtifacts();
    const X = [], y = [];
    for (const r of rows) {
      const rowArr = [];
      for (const feat of resolvedFeatureNames) {
        if (this.numericCols.includes(feat)) {
          rowArr.push(r[feat]);
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

    const resolveNumeric = (key, fallback) => {
      let val = fallback;
      if (!Number.isFinite(val)) val = 0;
      if (!alreadyForward) {
        if (key === "last_winner") {
          if (val === 0 || val === 1) val = 1 - val;
        } else if (this._flipOnReverse.has(key)) {
          val = -val;
        }
      }
      numeric[key] = val;
      vectorInput[key] = val;
    };

    const rank1 = Number.isFinite(player1Snapshot?.rank)
      ? player1Snapshot.rank
      : (alreadyForward ? match.rank1 : match.rank2);
    const rank2 = Number.isFinite(player2Snapshot?.rank)
      ? player2Snapshot.rank
      : (alreadyForward ? match.rank2 : match.rank1);
    const rankDiff =
      Number.isFinite(rank1) && Number.isFinite(rank2) ? rank2 - rank1 : match.numeric.rank_diff;
    resolveNumeric("rank_diff", rankDiff);

    const pts1 = alreadyForward ? match.pts1 : match.pts2;
    const pts2 = alreadyForward ? match.pts2 : match.pts1;
    const ptsDiff = Number.isFinite(pts1) && Number.isFinite(pts2)
      ? pts1 - pts2
      : match.numeric.pts_diff;
    resolveNumeric("pts_diff", ptsDiff);

    resolveNumeric("odd_diff", match.numeric.odd_diff);
    resolveNumeric("h2h_advantage", match.numeric.h2h_advantage);
    resolveNumeric("last_winner", match.numeric.last_winner);
    resolveNumeric("surface_winrate_adv", match.numeric.surface_winrate_adv);
    resolveNumeric("year", SCENARIO_YEAR);

    const categorical = {};
    for (const col of this.categoricalCols) {
      const val = match.categorical[col] ?? "";
      categorical[col] = val;
      vectorInput[col] = val;
    }

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
      vectorInput
    };
  }

  _featureNamesFromArtifacts() {
    const featureNames = [...this.numericCols];
    for (const col of this.categoricalCols) {
      const levels = this.catLevels[col] || [];
      for (const lvl of levels) featureNames.push(`${col}__${lvl}`);
    }
    return featureNames;
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

  _validateGruFeatures(headers, featureList) {
    const missing = featureList.filter((f) => !headers.includes(f));
    if (missing.length > 0) {
      throw new Error(`Missing feature(s) for sequence model: ${missing.join(", ")}`);
    }
  }

  _computeScalerForRows(rows, featureList) {
    const mean = {}, std = {};
    featureList.forEach((f) => {
      let sum = 0, sumSq = 0, count = 0;
      for (const r of rows) {
        const v = this._toNumber(r[f]);
        if (!Number.isFinite(v)) {
          throw new Error(`Missing feature ${f} in dataset`);
        }
        sum += v;
        sumSq += v * v;
        count += 1;
      }
      const mu = sum / Math.max(1, count);
      const variance = Math.max(0, sumSq / Math.max(1, count) - mu * mu);
      mean[f] = mu;
      std[f] = Math.sqrt(variance);
    });
    return { mean, std };
  }

  _normalizeRowsForGru(rows, mean, std, featureList) {
    return rows.map((row) => {
      const normalized = { ...row };
      featureList.forEach((f) => {
        const raw = this._toNumber(row[f]);
        if (!Number.isFinite(raw)) {
          throw new Error(`Missing feature ${f} in dataset`);
        }
        const mu = mean[f] ?? 0;
        const sigma = std[f] ?? 1;
        const val = sigma === 0 ? 0 : (raw - mu) / sigma;
        normalized[f] = Number.isFinite(val) ? val : 0;
      });
      normalized.y = Math.round(row[this.labelCol]);
      if (!Number.isFinite(normalized.y)) normalized.y = 0;
      return normalized;
    });
  }

  _orientFeaturesForPlayer(row, playerName, featureList, mean, std) {
    const p1 = (row.Player_1 || row.player1 || row.player || "").toString().trim();
    const p2 = (row.Player_2 || row.player2 || "").toString().trim();
    const isP1 = p1 === playerName;
    const isP2 = p2 === playerName;
    if (!isP1 && !isP2) return null;
    const flip = !isP1 && isP2;
    const values = featureList.map((f) => {
      let raw = this._toNumber(row[f]);
      if (!Number.isFinite(raw)) raw = 0;
      if (flip && this._flipOnReverse.has(f)) raw = -raw;
      const mu = mean[f] ?? 0;
      const sigma = std[f] ?? 1;
      const norm = sigma === 0 ? 0 : (raw - mu) / sigma;
      return Number.isFinite(norm) ? norm : 0;
    });
    const date = row.date || row.match_date || row.Date || row.rawDate || "";
    const timestamp = Number.isFinite(row.timestamp) ? row.timestamp : this._parseDate(date);
    return { values, date, timestamp };
  }

  _collectPlayerHistory(playerName, featureList, mean, std) {
    const rows = this.sequenceRowsRaw || this.sequenceRows || [];
    const history = [];
    for (const row of rows) {
      const oriented = this._orientFeaturesForPlayer(row, playerName, featureList, mean, std);
      if (!oriented) continue;
      history.push(oriented);
    }
    history.sort((a, b) => (a.timestamp || -Infinity) - (b.timestamp || -Infinity));
    return history;
  }

  _padSequence(history, seqLen, featureCount) {
    const padding = Math.max(0, seqLen - history.length);
    const padded = [];
    for (let i = 0; i < padding; i++) {
      padded.push(Array.from({ length: featureCount }, () => 0));
    }
    for (const h of history.slice(-seqLen)) {
      padded.push(h.values.slice());
    }
    return padded.slice(-seqLen);
  }
}
