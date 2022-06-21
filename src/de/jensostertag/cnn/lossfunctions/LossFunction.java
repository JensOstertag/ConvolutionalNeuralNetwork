package de.jensostertag.cnn.lossfunctions;

public interface LossFunction {
    BinaryCrossEntropy lossBinaryCrossEntropy = new BinaryCrossEntropy();
    MeanSquaredError lossMeanSquaredError = new MeanSquaredError();
    SquaredError lossSquaredError = new SquaredError();
    
    double function(double[] expected, double[] actual);
    double[] derivative(double[] expected, double[] actual);
}
