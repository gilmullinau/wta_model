// sequence-builder.js
// Builds fixed-length sequences for GRU models from feature-engineered rows.

function resolvePlayer(row) {
  return (
    (row &&
      (row.player || row.Player || row.player1 || row.player_1 || row.playerOne || row.Player_1)) ||
    ""
  ).toString().trim();
}

function resolveDate(row) {
  if (row && Number.isFinite(row.timestamp)) return row.timestamp;
  const raw = (
    row.date || row.Date || row.match_date || row.matchDate || row.datetime || row.timestamp || ""
  ).toString();
  const parsed = Date.parse(raw);
  if (Number.isFinite(parsed)) return parsed;
  return null;
}

function toNumber(value) {
  const num = typeof value === "number" ? value : Number.parseFloat(value);
  return Number.isFinite(num) ? num : NaN;
}

export function buildSequences(rows, seqLen, featureList) {
  if (!Array.isArray(rows)) throw new Error("rows must be an array");
  if (!Number.isInteger(seqLen) || seqLen <= 0) throw new Error("seqLen must be a positive integer");
  if (!Array.isArray(featureList) || featureList.length === 0) {
    throw new Error("featureList must be a non-empty array");
  }

  const featureIndexMap = {};
  featureList.forEach((f, idx) => {
    featureIndexMap[f] = idx;
  });

  const playerMatches = new Map();
  rows.forEach((row, idx) => {
    const player = resolvePlayer(row);
    if (!player) throw new Error(`Row ${idx} is missing player identifier`);
    const timestamp = resolveDate(row);
    if (!Number.isFinite(timestamp)) throw new Error(`Row ${idx} has invalid date`);

    const featureVector = featureList.map((key) => {
      const v = toNumber(row[key]);
      return Number.isFinite(v) ? v : NaN;
    });

    const match = {
      player,
      timestamp,
      features: featureVector,
      label: toNumber(row.y ?? row.label ?? row.target) >= 0.5 ? 1 : 0,
      rawDate: row.date || row.Date || "",
    };
    if (!playerMatches.has(player)) playerMatches.set(player, []);
    playerMatches.get(player).push(match);
  });

  let hasNaN = false;
  let paddingCount = 0;
  const X = [];
  const y = [];
  const sampleInfo = [];

  for (const [, matches] of playerMatches) {
    matches.sort((a, b) => a.timestamp - b.timestamp);
    matches.forEach((match, idx) => {
      const historyStart = Math.max(0, idx - seqLen);
      const history = matches.slice(historyStart, idx);
      const seq = Array.from({ length: seqLen }, () => Array(featureList.length).fill(0));
      const startOffset = seqLen - history.length;
      if (history.length < seqLen) paddingCount += 1;

      history.forEach((h, j) => {
        seq[startOffset + j] = h.features.map((v) => {
          if (!Number.isFinite(v)) {
            hasNaN = true;
            return 0;
          }
          return v;
        });
      });

      X.push(seq);
      y.push(match.label);
      sampleInfo.push({
        player: match.player,
        date: new Date(match.timestamp).toISOString().slice(0, 10),
        label: match.label,
      });
    });
  }

  const numSamples = X.length;
  const paddingPercent = numSamples === 0 ? 0 : (paddingCount / numSamples) * 100;

  return {
    X,
    y,
    meta: {
      seqLen,
      numFeatures: featureList.length,
      featureList: featureList.slice(),
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
