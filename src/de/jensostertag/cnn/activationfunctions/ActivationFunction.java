package de.jensostertag.cnn.activationfunctions;

public interface ActivationFunction {
    Relu activationRelu = new Relu();
    Sigmoid activationSigmoid = new Sigmoid();
    Step activationStep = new Step();
    
    double function(double x);
    double derivative(double x);
    double where(double y);
}
