package de.jensostertag.cnn.neuralnetwork;

import de.jensostertag.cnn.dataset.Dataset;
import de.jensostertag.cnn.lossfunctions.LossFunction;
import de.jensostertag.cnn.neuralnetwork.layers.ConvolutionalLayer;
import de.jensostertag.cnn.neuralnetwork.layers.Layer;

import java.util.ArrayList;
import java.util.Arrays;

public class CNN {
    protected ArrayList<Layer> layers = new ArrayList<>();
    
    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }
    
    public Object[] propagate(Object input) {
        Object[] output = new Object[this.layers.size() + 1];
        output[0] = input;
    
        for(int i = 0; i < this.layers.size(); i++)
            output[i + 1] = this.layers.get(i).propagate(output[i]);
    
        return output;
    }
    
    public double getAccuracy(Dataset testingDataset) {
        if(testingDataset.isValid()) {
            double accuracy = 1;
            
            for(int i = 0; i < testingDataset.getInputs().length; i++) {
                Object input = testingDataset.getInput(i);
                Object expectedOutput = testingDataset.getExpectedOutput(i);
                Object[] output = this.propagate(input);
                Object actualOutput = output[output.length - 1];
                
                if(expectedOutput instanceof double[] expected && actualOutput instanceof double[] actual) {
                    int[] expectedIndices = new int[0];
                    if(expected.length > 1) {
                        double highestExpected = expected[0];
                        for(int j = 1; j < expected.length; j++) {
                            if(expected[j] >= highestExpected) {
                                expectedIndices = new int[]{j};
                                highestExpected = expected[j];
                            } else {
                                int[] newExpectedIndices = new int[expectedIndices.length + 1];
                                for(int k = 0; k < expectedIndices.length; k++)
                                    newExpectedIndices[k] = expectedIndices[k];
                                newExpectedIndices[newExpectedIndices.length - 1] = j;
                                expectedIndices = newExpectedIndices;
                            }
                        }
                    }
                    
                    int[] actualIndices = new int[0];
                    if(actual.length > 1) {
                        double highestActual = actual[0];
                        for(int j = 1; j < actual.length; j++) {
                            if(actual[j] >= highestActual) {
                                actualIndices = new int[]{j};
                                highestActual = actual[j];
                            } else {
                                int[] newActualIndices = new int[actualIndices.length + 1];
                                for(int k = 0; k < actualIndices.length; k++)
                                    newActualIndices[k] = actualIndices[k];
                                newActualIndices[newActualIndices.length - 1] = j;
                                actualIndices = newActualIndices;
                            }
                        }
                    }
                    
                    if(!(Arrays.equals(expectedIndices, actualIndices)))
                        accuracy -= 1d / testingDataset.getInputs().length;
                } else
                    accuracy -= 1d / testingDataset.getInputs().length;
            }
            
            return accuracy;
        } else
            throw new IllegalArgumentException("Datasets are supposed to be validated");
    }
    
    public double train(Dataset trainingDataset, Dataset testingDataset, LossFunction lossFunction, int epochs) {
        if(trainingDataset.isValid() && testingDataset.isValid()) {
            double previousAccuracy = 0;
            double currentAccuracy = this.getAccuracy(testingDataset);
            double bestAccuracy = currentAccuracy;
            int iterationsCounter = 0;
            
            ArrayList<Layer> bestLayers = new ArrayList<>(this.layers.size());
            
            double currentLearningRate = Config.STARTING_LEARNING_RATE;
            
            for(int e = 0; e < epochs; e++) {
                if(iterationsCounter != 0) {
                    if(Math.abs(previousAccuracy - currentAccuracy) <= Config.ADJUST_LEARNING_RATE_DIFF)
                        currentLearningRate *= (1 + Config.ADJUST_LEARNING_RATE);
                    else
                        currentLearningRate /= (1 + Config.ADJUST_LEARNING_RATE);
                    
                    currentLearningRate = Math.min(Config.LEARNING_RATE_MAX, currentLearningRate);
                    currentLearningRate = Math.max(Config.LEARNING_RATE_MIN, currentLearningRate);
                }
                
                double loss = 0;
                
                for(int i = 0; i < trainingDataset.getInputs().length; i++) {
                    Object input = trainingDataset.getInput(i);
                    Object expectedOutput = trainingDataset.getExpectedOutput(i);
                    Object[] output = this.propagate(input);
    
                    loss += lossFunction.function((double[])expectedOutput, (double[])output[output.length - 1]);
                    
                    Object gradient = lossFunction.derivative((double[]) expectedOutput, (double[]) output[output.length - 1]);
                    for(int j = this.layers.size() - 1; j >= 0; j--)
                        gradient = this.layers.get(j).backPropagate(gradient, currentLearningRate);
                }
                
                loss /= trainingDataset.getInputs().length;
                
                System.out.print("Training " + (100*(double)(e+1)/epochs) + "% complete with " + (e+1) + " Epochs, current Loss: " + loss);
                previousAccuracy = currentAccuracy;
                currentAccuracy = this.getAccuracy(testingDataset);
                System.out.print(", Accuracy: " + currentAccuracy);
                System.out.println();
                
                ConvolutionalLayer cl = (ConvolutionalLayer) this.layers.get(0);
                if(e >= 75)
                    System.out.println(Arrays.deepToString(cl.kernel));
    
                if(currentAccuracy > bestAccuracy) {
                    bestAccuracy = currentAccuracy;
                }
    
                iterationsCounter++;
            }
            
            System.out.println(bestAccuracy);
            return bestAccuracy;
        } else
            throw new IllegalStateException("Datasets are supposed to be validated");
    }
}
