package de.jensostertag.cnn.neuralnetwork.layers;

import de.jensostertag.cnn.activationfunctions.ActivationFunction;
import de.jensostertag.cnn.activationfunctions.LayerActivation;
import de.jensostertag.cnn.neuralnetwork.Config;

public class FullyConnectedLayer implements Layer {
    protected final int INPUT_LENGTH;
    protected final int OUTPUT_LENGTH;
    protected double[][] weights;
    protected final ActivationFunction activationFunction;
    
    public FullyConnectedLayer(int INPUT_LENGTH, int OUTPUT_LENGTH, ActivationFunction activationFunction) {
        this.INPUT_LENGTH = INPUT_LENGTH;
        this.OUTPUT_LENGTH = OUTPUT_LENGTH;
        this.weights = new double[this.INPUT_LENGTH + 1][this.OUTPUT_LENGTH];
        this.activationFunction = activationFunction;
        for(int i = 0; i < this.weights.length; i++)
            for(int j = 0; j < this.weights[i].length; j++)
                this.weights[i][j] = Config.DEFAULT_WEIGHT_MIN + Math.random() * (Config.DEFAULT_WEIGHT_MAX - Config.DEFAULT_WEIGHT_MIN);
    }
    
    @Override
    public Object propagate(Object input) {
        if(input instanceof double[] layerInput) {
            if(layerInput.length == this.INPUT_LENGTH) {
                double[] output = new double[this.OUTPUT_LENGTH];
                
                for(int i = 0; i < this.weights.length; i++) {
                    for(int j = 0; j < this.weights[i].length; j++) {
                        double inputValue = 1;
                        if(i != this.weights.length - 1)
                            inputValue = layerInput[i];
                        double weight = this.weights[i][j];
                        
                        output[j] += inputValue * weight;
                    }
                }
                
                return LayerActivation.activate(this.activationFunction, output);
            } else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a Double Array");
    }
    
    @Override
    public double[] mistakes(Object previousMistakes, Object layerOutput) {
        if(previousMistakes instanceof double[] mistakes && layerOutput instanceof double[] output) {
            if(mistakes.length == this.OUTPUT_LENGTH && output.length == this.OUTPUT_LENGTH) {
                double[] newMistakes = new double[this.INPUT_LENGTH];
                
                for(int i = 0; i < this.weights.length - 1; i++) {
                    for(int j = 0; j < mistakes.length; j++) {
                        double outputValue = output[j];
                        double derivative = this.activationFunction.derivative(this.activationFunction.where(outputValue));
                        double mistake = mistakes[j];
                        double weight = this.weights[i][j];
                        
                        newMistakes[i] += derivative * mistake * weight;
                    }
                }
                
                return newMistakes;
            } else
                throw new IllegalArgumentException("PreviousMistakes or LayerOutput is not of correct Size");
        } else
            throw new IllegalArgumentException("PreviousMistakes and LayerOutput are supposed to be a Double Array");
    }
    
    public double[] outputMistakes(Object expectedOutput, Object layerOutput) {
        if(expectedOutput instanceof double[] expected && layerOutput instanceof double[] output) {
            if(expected.length == this.OUTPUT_LENGTH && output.length == this.OUTPUT_LENGTH) {
                double[] newMistakes = new double[this.OUTPUT_LENGTH];
                
                for(int i = 0; i < newMistakes.length; i++) {
                    double outputValue = output[i];
                    double derivative = this.activationFunction.derivative(this.activationFunction.where(outputValue));
                    double expectedValue = expected[i];
                    
                    newMistakes[i] = derivative * (outputValue - expectedValue);
                }
            
                return newMistakes;
            } else
                throw new IllegalArgumentException("ExpectedOutput or LayerOutput is not of correct Size");
        } else
            throw new IllegalArgumentException("ExpectedOutput and LayerOutput are supposed to be a Double Array");
    }
    
    @Override
    public void optimizeWeights(Object previousMistakes, Object layerOutput, double learningRate) {
        if(previousMistakes instanceof double[] mistakes && layerOutput instanceof double[] output) {
            if(mistakes.length == this.OUTPUT_LENGTH && output.length == this.OUTPUT_LENGTH) {
                double[][] newWeights = new double[this.weights.length][];
                for(int i = 0; i < newWeights.length; i++)
                    newWeights[i] = this.weights[i].clone();
    
                for(int i = 0; i < this.weights.length - 1; i++) {
                    for(int j = 0; j < mistakes.length; j++) {
                        double outputValue = output[j];
                        double mistake = mistakes[j];
            
                        double deltaWeight = learningRate * outputValue * mistake;
                        newWeights[i][j] += deltaWeight;
                    }
                }
                
                this.weights = newWeights;
            } else
                throw new IllegalArgumentException("There are too many or too few Mistakes or Outputs");
        } else
            throw new IllegalArgumentException("PreviousMistakes and LayerOutput are supposed to be a Double Array");
    }
}
