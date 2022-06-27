package de.jensostertag.cnn.neuralnetwork.layers;

import de.jensostertag.cnn.neuralnetwork.util.Matrices;
import de.jensostertag.cnn.neuralnetwork.util.pooling.Pooling;
import de.jensostertag.cnn.neuralnetwork.util.pooling.PoolingType;

public class PoolingLayer implements Layer {
    private final int INPUT_CHANNELS;
    private final int INPUT_WIDTH;
    private final int INPUT_HEIGHT;
    private final Pooling POOLING;
    
    private Object input;
    
    public PoolingLayer(int INPUT_CHANNELS, int INPUT_WIDTH, int INPUT_HEIGHT, Pooling POOLING) {
        this.INPUT_CHANNELS = INPUT_CHANNELS;
        this.INPUT_WIDTH = INPUT_WIDTH;
        this.INPUT_HEIGHT = INPUT_HEIGHT;
        this.POOLING = POOLING;
        
        if(this.INPUT_WIDTH % this.POOLING.POOLING_SIZE != 0 || this.INPUT_HEIGHT % this.POOLING.POOLING_SIZE != 0)
            throw new IllegalArgumentException("Input Width and Height must be a Multiple of " + this.POOLING.POOLING_SIZE);
    }
    
    @Override
    public double[][][] propagate(Object input) {
        if(input instanceof double[][][] layerInput) {
            this.input = input;
            
            if(Matrices.validateSize(layerInput, this.INPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH)) {
                int ps = this.POOLING.POOLING_SIZE;
                double[][][] output = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT / ps][this.INPUT_WIDTH / ps];
                
                for(int i = 0; i < output.length; i++) {
                    for(int j = 0; j < output[i].length; j++) {
                        for(int k = 0; k < output[i][j].length; k++) {
                            double[][] subMatrix = Matrices.subMatrix(layerInput[i], j * ps, k * ps, ps, ps);
                            if(this.POOLING.POOLING_TYPE == PoolingType.MAX) {
                                int[] highestIndex = Matrices.highestValue(subMatrix);
                                output[i][j][k] = subMatrix[highestIndex[0]][highestIndex[1]];
                            } else if(this.POOLING.POOLING_TYPE == PoolingType.MEAN) {
                                double sum = Matrices.sum(subMatrix);
                                output[i][j][k] = sum / (ps * ps);
                            }
                        }
                    }
                }
                
                return output;
            } else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a 3-Dimensional Double Array");
    }
    
    @Override
    public double[][][] backPropagate(Object d_L_d_Y, double learningRate) {
        if(d_L_d_Y instanceof double[][][] gradient && this.input instanceof double[][][] layerInput) {
            int ps = this.POOLING.POOLING_SIZE;
            if(Matrices.validateSize(gradient, this.INPUT_CHANNELS, this.INPUT_HEIGHT / ps, this.INPUT_WIDTH / ps)) {
                if(Matrices.validateSize(layerInput, this.INPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH)) {
                    double[][][] output = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
                    for(int i = 0; i < gradient.length; i++) {
                        for(int j = 0; j < gradient[i].length; j++) {
                            for(int k = 0; k < gradient[i][j].length; k++) {
                                double[][] subMatrix = Matrices.subMatrix(layerInput[i], j * ps, k * ps, ps, ps);
                                if(this.POOLING.POOLING_TYPE == PoolingType.MAX) {
                                    int[] highestIndex = Matrices.highestValue(subMatrix);
                                    output[i][j * ps + highestIndex[0]][k * ps + highestIndex[1]] = gradient[i][j][k];
                                } else if(this.POOLING.POOLING_TYPE == PoolingType.MEAN) {
                                    for(int l = 0; l < ps; l++)
                                        for(int m = 0; m < ps; m++)
                                            output[i][j * ps + l][k * ps + m] = gradient[i][j][k];
                                }
                            }
                        }
                    }
                    
                    return output;
                } else
                    throw new IllegalArgumentException("Input is not of correct Size");
            } else
                throw new IllegalArgumentException("Gradient is not of correct Size");
        } else
            throw new IllegalArgumentException("Gradient and Input are supposed to be 3-Dimensional Double Arrays");
    }
}
