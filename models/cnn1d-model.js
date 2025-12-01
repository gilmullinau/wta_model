const tf = window.tf;

export function buildCNNModel(inputShape, config = {}) {
  const cfg = {
    filters: Number.isFinite(config.filters) ? config.filters : 32,
    kernelSize: Number.isFinite(config.kernelSize) ? config.kernelSize : 3,
    denseUnits: Number.isFinite(config.denseUnits) ? config.denseUnits : 32,
    learningRate: typeof config.learningRate === "number" ? config.learningRate : 0.001,
  };

  const model = tf.sequential();

  model.add(tf.layers.conv1d({
    inputShape,
    filters: cfg.filters,
    kernelSize: cfg.kernelSize,
    activation: "relu",
    padding: "same",
  }));

  model.add(tf.layers.conv1d({
    filters: cfg.filters,
    kernelSize: cfg.kernelSize,
    activation: "relu",
    padding: "same",
  }));

  model.add(tf.layers.globalAveragePooling1d());
  model.add(tf.layers.dense({ units: cfg.denseUnits, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  const optimizer = tf.train.adam(cfg.learningRate || 0.001);
  model.compile({
    optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}
