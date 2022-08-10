package de.jensostertag.cnn.activationfunctions;

public interface ActivationFunction {
    Linear activationLinear = new Linear();
    Relu activationRelu = new Relu();
    Sigmoid activationSigmoid = new Sigmoid();
    Step activationStep = new Step();
    
    /**
     * Activation Function
     * @param x Input
     * @return Activation Function of Input
     */
    double function(double x);
    
    /**
     * Derivative of Activation Function
     * @param x Input
     * @return Derivative of Activation Function of Input
     */
    double derivative(double x);
    
    /**
     * Inverse of Activation Function
     * @param y Output
     * @return Input where Activation Function of Input is Output
     */
    double where(double y);
}
