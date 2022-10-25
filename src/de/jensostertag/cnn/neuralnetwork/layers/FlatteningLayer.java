package de.jensostertag.cnn.neuralnetwork.layers;

import de.jensostertag.cnn.neuralnetwork.util.Matrices;

public class FlatteningLayer implements Layer {
    private final int INPUT_CHANNELS;
    private final int INPUT_WIDTH;
    private final int INPUT_HEIGHT;
    
    public FlatteningLayer(int INPUT_CHANNELS, int INPUT_WIDTH, int INPUT_HEIGHT) {
        this.INPUT_CHANNELS = INPUT_CHANNELS;
        this.INPUT_WIDTH = INPUT_WIDTH;
        this.INPUT_HEIGHT = INPUT_HEIGHT;
    }
    
    @Override
    public double[] propagate(Object input) {
        if(input instanceof double[][][] layerInput) {
            if(Matrices.validateSize(layerInput, this.INPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH))
                return Matrices.flatten(layerInput);
            else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a 3-Dimensional Double Array");
    }
    
    @Override
    public double[][][] backPropagate(Object d_L_d_Y, double learningRate) {
        if(d_L_d_Y instanceof double[] gradient) {
            if(gradient.length == this.INPUT_CHANNELS * this.INPUT_HEIGHT * this.INPUT_WIDTH) {
                double[][][] output = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
                for(int i = 0; i < gradient.length; i++) {
                    int channel = i / (this.INPUT_HEIGHT * this.INPUT_WIDTH);
                    int row = i / this.INPUT_WIDTH - channel * this.INPUT_HEIGHT;
                    int column = i - this.INPUT_WIDTH * (this.INPUT_HEIGHT * channel + row);
                    output[channel][row][column] = gradient[i];
                }
                
                return output;
            } else
                throw new IllegalArgumentException("Gradient is not of correct Size");
        } else
            throw new IllegalArgumentException("Gradient is supposed to be a Double Array");
    }

    @Override
    public FlatteningLayer copy() {
        return new FlatteningLayer(this.INPUT_CHANNELS, this.INPUT_WIDTH, this.INPUT_HEIGHT);
    }
}
