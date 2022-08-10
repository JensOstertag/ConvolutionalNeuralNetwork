package de.jensostertag.cnn.activationfunctions;

public interface ActivationFunction {
    Linear activationLinear = new Linear();
    Relu activationRelu = new Relu();
    Sigmoid activationSigmoid = new Sigmoid();
    Step activationStep = new Step();
    
    double function(double x);
    double derivative(double x);
    double where(double y);
}
