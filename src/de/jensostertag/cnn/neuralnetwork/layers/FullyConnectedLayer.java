package de.jensostertag.cnn.neuralnetwork.layers;

import de.jensostertag.cnn.activationfunctions.ActivationFunction;
import de.jensostertag.cnn.activationfunctions.LayerActivation;
import de.jensostertag.cnn.neuralnetwork.Config;
import de.jensostertag.cnn.neuralnetwork.util.Matrices;

public class FullyConnectedLayer implements Layer {
    private final int INPUT_LENGTH;
    private final int OUTPUT_LENGTH;
    private final ActivationFunction activationFunction;
    public double[][] weights;
    private double[] biases;
    
    private Object input;
    
    public FullyConnectedLayer(int INPUT_LENGTH, int OUTPUT_LENGTH, ActivationFunction activationFunction) {
        this.INPUT_LENGTH = INPUT_LENGTH;
        this.OUTPUT_LENGTH = OUTPUT_LENGTH;
        this.weights = Matrices.randomMatrix(this.INPUT_LENGTH, this.OUTPUT_LENGTH, Config.DEFAULT_WEIGHT_MIN, Config.DEFAULT_WEIGHT_MAX);
        this.biases = Matrices.singleValueMatrix(this.OUTPUT_LENGTH, 0);
        this.activationFunction = activationFunction;
    }
    
    @Override
    public double[] propagate(Object input) {
        if(input instanceof double[] layerInput) {
            this.input = input;
            
            if(layerInput.length == this.INPUT_LENGTH) {
                double[] output = new double[this.OUTPUT_LENGTH];
                output = Matrices.flatten(Matrices.add(Matrices.multiply(Matrices.asMatrix(layerInput), this.weights), Matrices.asMatrix(this.biases)));
                return (double[]) LayerActivation.activate(this.activationFunction, output);
            } else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a Double Array");
    }
    
    @Override
    public double[] backPropagate(Object d_L_d_Y, double learningRate) {
        if(d_L_d_Y instanceof double[] gradient && this.input instanceof double[] layerInput) {
            if(gradient.length == this.OUTPUT_LENGTH && layerInput.length == this.INPUT_LENGTH) {
                double[] net = Matrices.flatten(Matrices.add(Matrices.multiply(Matrices.asMatrix(layerInput), this.weights), Matrices.asMatrix(this.biases)));
                double[] d_Y_d_net = (double[]) LayerActivation.derive(this.activationFunction, net);
                double[] d_L_d_net = new double[this.OUTPUT_LENGTH];
                for(int i = 0; i < d_L_d_net.length; i++)
                    d_L_d_net[i] = gradient[i] * activationFunction.derivative(net[i]);
                double[][] d_net_d_W = Matrices.transpose(Matrices.asMatrix(layerInput));
                double[][] d_net_d_X = this.weights;
                
                double[][] d_L_d_W = Matrices.multiply(d_net_d_W, Matrices.asMatrix(d_L_d_net));
                double[] d_L_d_B = d_L_d_net;
                
                double[] d_L_d_X = Matrices.asVector(Matrices.multiply(Matrices.asMatrix(d_L_d_net), Matrices.transpose(d_net_d_X)));
                
                this.weights = Matrices.add(Matrices.multiplyConstant(d_L_d_W, -learningRate), this.weights);
                this.biases = Matrices.add(Matrices.multiplyConstant(d_L_d_B, -learningRate), this.biases);
                
                return d_L_d_X;
            } else
                throw new IllegalArgumentException("Gradient is not of correct Size");
        } else
            throw new IllegalArgumentException("Gradient is supposed to be a Double Array");
    }
    
    public double[] outputGradient(Object expectedOutput, Object actualOutput) {
        if(expectedOutput instanceof double[] expected && actualOutput instanceof double[] actual) {
            if(expected.length == this.OUTPUT_LENGTH && actual.length == this.OUTPUT_LENGTH) {
                double[] loss = new double[this.OUTPUT_LENGTH];
                for(int i = 0; i < loss.length; i++)
                    loss[i] = actual[i] - expected[i];
                    
                return loss;
            } else
                throw new IllegalArgumentException("ExpectedOutput is not of correct Size");
        } else
            throw new IllegalArgumentException("ExpectedOutput is supposed to be a Double Array");
    }
}
