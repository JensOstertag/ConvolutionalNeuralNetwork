package de.jensostertag.cnn.lossfunctions;

import de.jensostertag.cnn.neuralnetwork1.layers.Layer;
import de.jensostertag.cnn.neuralnetwork1.util.Matrices;

public class BinaryCrossEntropy implements LossFunction {
    @Override
    public double function(double[] expected, double[] actual) {
        int length = Matrices.getSameSize(expected, actual)[0];
        double loss = 0;
        for(int i = 0; i < length; i++)
            loss += (1d/length) * (-expected[i] * Math.log(actual[i]) - (1 - expected[i]) * Math.log(1 - actual[i]));
        return loss;
    }
    
    @Override
    public double[] derivative(double[] expected, double[] actual) {
        int length = Matrices.getSameSize(expected, actual)[0];
        double[] output = new double[length];
    
        for(int i = 0; i < output.length; i++)
            output[i] = ((1 - expected[i]) / (1 - actual[i]) - expected[i] / actual[i]) / length;
    
        return output;
    }
}
