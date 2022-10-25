package de.jensostertag.cnn.neuralnetwork.layers;

public interface Layer {
    Object propagate(Object input);
    Object backPropagate(Object d_L_d_Y, double learningRate);
    Layer copy();
}
