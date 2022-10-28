package de.jensostertag.cnn.activationfunctions;

public class Sigmoid implements ActivationFunction {
    @Override
    public double function(double x) {
        return 1d / (1d + Math.exp(-x));
    }
    
    @Override
    public double derivative(double x) {
        return function(x) * (1 - function(x));
    }
    
    @Override
    public double where(double y) {
        return - Math.log(1/y - 1);
    }
}
