package de.jensostertag.cnn.lossfunctions;

public interface LossFunction {
    BinaryCrossEntropy lossBinaryCrossEntropy = new BinaryCrossEntropy();
    MeanSquaredError lossMeanSquaredError = new MeanSquaredError();
    SquaredError lossSquaredError = new SquaredError();
    
    /**
     * Loss Function
     * @param expected Expected Value
     * @param actual Actual Value
     * @return Loss between actual and expected Value
     */
    double function(double[] expected, double[] actual);
    
    /**
     * Derivative of Loss Function
     * @param expected Expected Value
     * @param actual Actual Value
     * @return Derivative of Loss between actual and expected Value
     */
    double[] derivative(double[] expected, double[] actual);
}
