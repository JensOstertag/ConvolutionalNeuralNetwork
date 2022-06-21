package de.jensostertag.cnn.lossfunctions;

import de.jensostertag.cnn.neuralnetwork1.util.Matrices;

public class SquaredError implements LossFunction {
    @Override
    public double function(double[] expected, double[] actual) {
        int length = Matrices.getSameSize(expected, actual)[0];
        double loss = 0;
        for(int i = 0; i < length; i++) {
            loss += Math.pow(expected[i] - actual[i], 2);
        }
        return loss;
    }
    
    @Override
    public double[] derivative(double[] expected, double[] actual) {
        int length = Matrices.getSameSize(expected, actual)[0];
        double[] output = new double[length];
        
        for(int i = 0; i < output.length; i++)
            output[i] = 2 * (actual[i] - expected[i]);
        
        return output;
    }
}
