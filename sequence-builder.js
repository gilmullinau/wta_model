// sequence-builder.js
// Builds fixed-length sequences for GRU models from feature-engineered rows.

function toNumber(value) {
  const num = typeof value === "number" ? value : Number.parseFloat(value);
  return Number.isFinite(num) ? num : NaN;
}

function resolveTimestamp(row) {
  if (Number.isFinite(row.timestamp)) return row.timestamp;
  const raw = (row.match_date || row.date || row.Date || row.rawDate || "").toString();
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function orientFeatures(row, featureList, mean, std, flipOnReverse, toPlayer2 = false) {
  return featureList.map((f) => {
    let raw = toNumber(row[f]);
    if (!Number.isFinite(raw)) raw = 0;
    if (toPlayer2 && flipOnReverse.has(f)) raw = -raw;
    const mu = mean[f] ?? 0;
    const sigma = std[f] ?? 1;
    const norm = sigma === 0 ? 0 : (raw - mu) / sigma;
    return Number.isFinite(norm) ? norm : 0;
  });
}

export function buildSequences(rows, seqLen, featureList, opts = {}) {
  if (!Array.isArray(rows)) throw new Error("rows must be an array");
  if (!Number.isInteger(seqLen) || seqLen <= 0) throw new Error("seqLen must be a positive integer");
  if (!Array.isArray(featureList) || featureList.length === 0) {
    throw new Error("featureList must be a non-empty array");
  }
  const mean = opts.mean || {};
  const std = opts.std || {};
  const flipOnReverse = new Set(opts.flipOnReverse || []);

  const featureIndexMap = {};
  featureList.forEach((f, idx) => {
    featureIndexMap[`p1_${f}`] = idx;
    featureIndexMap[`p2_${f}`] = idx + featureList.length;
  });

  const sorted = rows
    .map((row, idx) => ({
      row,
      idx,
      player1: (row.Player_1 || row.player1 || "").toString().trim(),
      player2: (row.Player_2 || row.player2 || "").toString().trim(),
      timestamp: resolveTimestamp(row),
    }))
    .filter((r) => r.player1 && r.player2 && Number.isFinite(r.timestamp))
    .sort((a, b) => a.timestamp - b.timestamp);

  const history = new Map();
  let hasNaN = false;
  let paddingCount = 0;
  const X = [];
  const y = [];
  const sampleInfo = [];

  for (const entry of sorted) {
    const { row, player1, player2, timestamp } = entry;
    const labelVal = toNumber(row.y ?? row.label ?? row.target);
    if (!Number.isFinite(labelVal)) continue;
    const label = labelVal >= 0.5 ? 1 : 0;

    const hist1 = history.get(player1) || [];
    const hist2 = history.get(player2) || [];

    const pad1 = Math.max(0, seqLen - hist1.length);
    const pad2 = Math.max(0, seqLen - hist2.length);
    if (pad1 > 0 || pad2 > 0) paddingCount += 1;

    const padded1 = Array.from({ length: pad1 }, () => Array(featureList.length).fill(0)).concat(hist1.slice(-seqLen));
    const padded2 = Array.from({ length: pad2 }, () => Array(featureList.length).fill(0)).concat(hist2.slice(-seqLen));
    const trimmed1 = padded1.slice(-seqLen);
    const trimmed2 = padded2.slice(-seqLen);

    const seq = trimmed1.map((step, idx) => {
      const v1 = step;
      const v2 = trimmed2[idx] || Array(featureList.length).fill(0);
      const merged = v1.concat(v2);
      if (merged.some((v) => !Number.isFinite(v))) hasNaN = true;
      return merged.map((v) => (Number.isFinite(v) ? v : 0));
    });

    X.push(seq);
    y.push(label);
    sampleInfo.push({
      player1,
      player2,
      date: new Date(timestamp).toISOString().slice(0, 10),
      label,
    });

    const featuresP1 = orientFeatures(row, featureList, mean, std, flipOnReverse, false);
    const featuresP2 = orientFeatures(row, featureList, mean, std, flipOnReverse, true);
    if (!history.has(player1)) history.set(player1, []);
    if (!history.has(player2)) history.set(player2, []);
    history.get(player1).push(featuresP1);
    history.get(player2).push(featuresP2);
  }

  const numSamples = X.length;
  const paddingPercent = numSamples === 0 ? 0 : (paddingCount / numSamples) * 100;

  return {
    X,
    y,
    meta: {
      seqLen,
      numFeatures: featureList.length * 2,
      featureList: featureList.flatMap((f) => [`p1_${f}`, `p2_${f}`]),
      featureIndexMap,
      sampleInfo,
    },
    stats: {
      numSamples,
      paddingPercent,
      hasNaN,
    }
  };
}
