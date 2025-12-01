// gru-model.js — GRU sequence model for match outcome prediction
const tf = window.tf;

export class GruModel {
  constructor(config = {}) {
    this.config = {
      units: Number.isFinite(config.units) ? config.units : 64,
      dropout: typeof config.dropout === "number" ? config.dropout : 0.2,
      denseUnits: Number.isFinite(config.denseUnits) ? config.denseUnits : 32,
      lr: typeof config.lr === "number" ? config.lr : 0.001,
    };
    this.model = null;
    this.modelKey = config.modelKey || "wta-gru-v1";
    this.metaKey = `${this.modelKey}-meta`;
    this.metadata = config.metadata || null;
  }

  build(inputShape) {
    if (!inputShape || inputShape.length !== 2) {
      throw new Error("GRU build requires inputShape [seqLen, numFeatures]");
    }
    if (this.model) this.dispose();
    this.model = tf.sequential();
    this.model.add(tf.layers.gru({
      units: this.config.units,
      returnSequences: false,
      inputShape,
      kernelInitializer: "glorotUniform",
      recurrentInitializer: "orthogonal",
      recurrentDropout: 0,
      dropout: 0,
    }));
    this.model.add(tf.layers.dropout({ rate: this.config.dropout }));
    this.model.add(tf.layers.dense({ units: this.config.denseUnits, activation: "relu" }));
    this.model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
    this.model.compile({
      optimizer: tf.train.adam(this.config.lr),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });
  }

  async train(X_train, y_train, X_val, y_val, { epochs = 6, batchSize = 128, onEpochEnd = null } = {}) {
    if (!this.model) throw new Error("Model not built. Call build() first.");
    const callbacks = {
      onEpochEnd: async (epoch, logs) => {
        if (onEpochEnd) onEpochEnd(epoch, logs);
        await tf.nextFrame();
      }
    };
    return await this.model.fit(X_train, y_train, {
      epochs,
      batchSize,
      validationData: X_val && y_val ? [X_val, y_val] : null,
      callbacks,
      shuffle: true,
    });
  }

  async evaluate(X_test, y_test) {
    if (!this.model) throw new Error("Model not built or loaded.");
    const evalOut = await this.model.evaluate(X_test, y_test, { batchSize: 256 });
    const [lossTensor, accTensor] = evalOut;
    const loss = (await lossTensor.data())[0];
    const acc = (await accTensor.data())[0];
    lossTensor.dispose();
    accTensor.dispose();
    return { loss, acc };
  }

  predictProba(batchX) {
    if (!this.model) throw new Error("Model not built or loaded.");
    return this.model.predict(batchX);
  }

  async predict(batchX) {
    const probs = this.predictProba(batchX);
    const data = await probs.data();
    probs.dispose();
    return data[0];
  }

  async confusionMatrix(X_test, y_test) {
    const probs = this.predictProba(X_test);
    const probsData = await probs.data();
    probs.dispose();
    const yTrue = Array.from(await y_test.data());
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const pred = probsData[i] >= 0.5 ? 1 : 0;
      if (pred === 1 && yTrue[i] === 1) tp++;
      else if (pred === 0 && yTrue[i] === 0) tn++;
      else if (pred === 1 && yTrue[i] === 0) fp++;
      else fn++;
    }
    return { tp, tn, fp, fn };
  }

  async save(key = this.modelKey) {
    if (!this.model) throw new Error("Model not built.");
    const saveKey = key || this.modelKey;
    await this.model.save(`localstorage://${saveKey}`);
    const metaPayload = {
      config: this.config,
      metadata: this.metadata || null,
    };
    localStorage.setItem(`${saveKey}_meta`, JSON.stringify(metaPayload));
  }

  static async load(key = "wta-gru-v1") {
    const model = await tf.loadLayersModel(`localstorage://${key}`);
    const rawMeta = localStorage.getItem(`${key}_meta`);
    let metaPayload = { config: {}, metadata: null };
    try {
      metaPayload = rawMeta ? JSON.parse(rawMeta) : metaPayload;
    } catch (err) {
      console.warn("Failed to parse GRU metadata", err);
    }
    const instance = new GruModel({ ...metaPayload.config, modelKey: key, metadata: metaPayload.metadata });
    instance.model = model;
    return instance;
  }

  setMetadata(metadata) {
    this.metadata = metadata || null;
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    if (tf?.engine) {
      tf.engine().startScope();
      tf.engine().disposeVariables();
      tf.engine().endScope();
    }
  }
}
