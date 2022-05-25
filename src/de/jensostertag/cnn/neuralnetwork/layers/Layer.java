package de.jensostertag.cnn.neuralnetwork.layers;

public interface Layer {
    Object propagate(Object input);
    Object mistakes(Object previousMistakes, Object layerOutput);
    void optimizeWeights(Object previousMistakes, Object layerOutput, double learningRate);
}
