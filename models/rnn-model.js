export function buildRNNModel(inputShape, config = {}) {
  const tf = window.tf;
  const model = tf.sequential();
  const units = Number.isFinite(config.units) ? config.units : 64;
  const denseUnits = Number.isFinite(config.denseUnits) ? config.denseUnits : 32;
  const dropout = Number.isFinite(config.dropout) ? config.dropout : 0.2;
  const learningRate = Number.isFinite(config.learningRate) ? config.learningRate : 0.001;

  model.add(tf.layers.simpleRNN({
    units,
    returnSequences: false,
    inputShape,
  }));
  model.add(tf.layers.dropout({ rate: dropout }));
  model.add(tf.layers.dense({ units: denseUnits, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
