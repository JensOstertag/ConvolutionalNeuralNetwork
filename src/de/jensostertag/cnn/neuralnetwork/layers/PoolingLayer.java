package de.jensostertag.cnn.neuralnetwork.layers;

import de.jensostertag.cnn.neuralnetwork.util.pooling.Pooling;
import de.jensostertag.cnn.neuralnetwork.util.pooling.PoolingType;

import java.util.Arrays;

public class PoolingLayer implements Layer {
    private final int INPUT_CHANNELS;
    private final int INPUT_WIDTH;
    private final int INPUT_HEIGHT;
    private final Pooling POOLING;
    
    public PoolingLayer(int INPUT_CHANNELS, int INPUT_WIDTH, int INPUT_HEIGHT, Pooling POOLING) {
        this.INPUT_CHANNELS = INPUT_CHANNELS;
        this.INPUT_WIDTH = INPUT_WIDTH;
        this.INPUT_HEIGHT = INPUT_HEIGHT;
        this.POOLING = POOLING;
        
        if(this.INPUT_WIDTH % this.POOLING.POOLING_SIZE != 0 || this.INPUT_HEIGHT % this.POOLING.POOLING_SIZE != 0)
            throw new IllegalArgumentException("Input Width and Height must be a Multiple of " + this.POOLING.POOLING_SIZE);
    }
    
    @Override
    public Object propagate(Object input) {
        if(input instanceof double[][][] layerInput) {
            if(layerInput.length == this.INPUT_CHANNELS) {
                int outputHeight = this.INPUT_HEIGHT / this.POOLING.POOLING_SIZE;
                int outputWidth = this.INPUT_WIDTH / this.POOLING.POOLING_SIZE;
                
                double[][][] output = new double[this.INPUT_CHANNELS][outputHeight][outputWidth];
                if(this.POOLING.POOLING_TYPE == PoolingType.MAX)
                    for(double[][] outputMatrix : output)
                        for(double[] outputVector : outputMatrix)
                            Arrays.fill(outputVector, Double.NEGATIVE_INFINITY);
                
                for(int i = 0; i < layerInput.length; i++) {
                    if(layerInput[i].length == this.INPUT_HEIGHT) {
                        for(int j = 0; j < layerInput[i].length; j++) {
                            if(layerInput[i][j].length == this.INPUT_WIDTH) {
                                for(int k = 0; k < layerInput[i][j].length; k++) {
                                    int outputRow = j / this.POOLING.POOLING_SIZE;
                                    int outputCol = k / this.POOLING.POOLING_SIZE;
                                    
                                    if(this.POOLING.POOLING_TYPE == PoolingType.MAX) {
                                        /*
                                         *  Max-Pooling
                                         */
                                        if(output[i][outputRow][outputCol] < layerInput[i][j][k])
                                            output[i][outputRow][outputCol] = layerInput[i][j][k];
                                    } else if(this.POOLING.POOLING_TYPE == PoolingType.MEAN) {
                                        /*
                                         *  Mean-Pooling
                                         */
                                        double value = layerInput[i][j][k];
                                        
                                        int poolingRow = j % this.POOLING.POOLING_SIZE;
                                        int poolingCol = k % this.POOLING.POOLING_SIZE;
                                        
                                        int nthValue = poolingRow * this.POOLING.POOLING_SIZE + poolingCol + 1;
                                        
                                        double currentMean = output[i][outputRow][outputCol];
                                        output[i][outputRow][outputCol] = (currentMean * (nthValue - 1) + value) / nthValue;
                                    }
                                }
                            } else
                                throw new IllegalArgumentException("Input is not of correct Size");
                        }
                    } else
                        throw new IllegalArgumentException("Input is not of correct Size");
                }
                
                return output;
            } else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a 3-Dimensional Double Array");
    }
    
    @Override
    public Object mistakes(Object previousMistakes, Object layerOutput) {
        if(previousMistakes instanceof double[][][] mistakes && layerOutput instanceof double[][][] output) {
            if(mistakes.length == this.INPUT_CHANNELS && output.length == this.INPUT_CHANNELS) {
                for(int i = 0; i < mistakes.length; i++) {
                    if(mistakes[i].length == this.POOLING.POOLING_SIZE && output[i].length == this.POOLING.POOLING_SIZE) {
                        for(int j = 0; j < mistakes[i].length; j++) {
                            if(mistakes[i][j].length != this.POOLING.POOLING_SIZE && output[i][j].length != this.POOLING.POOLING_SIZE)
                                throw new IllegalArgumentException("PreviousMistakes or LayerOutput is not of correct Size");
                        }
                    } else
                        throw new IllegalArgumentException("PreviousMistakes or LayerOutput is not of correct Size");
                }
                    
                double[][][] newMistakes = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
    
                for(int i = 0; i < newMistakes.length; i++)
                    for(int j = 0; j < newMistakes[i].length; j++)
                        for(int k = 0; k < newMistakes[i][j].length; k++)
                            newMistakes[i][j][k] = mistakes[i][j / this.POOLING.POOLING_SIZE][k / this.POOLING.POOLING_SIZE];
    
                return newMistakes;
            } else
                throw new IllegalArgumentException("PreviousMistakes or LayerOutput is not of correct Size");
        } else
            throw new IllegalArgumentException("PreviousMistakes and LayerOutput are supposed to be a 3-Dimensional Double Array");
    }
    
    @Override
    public void optimizeWeights(Object previousMistakes, Object layerOutput, double learningRate) {}
}
