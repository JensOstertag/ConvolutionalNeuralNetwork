package de.jensostertag.cnn.activationfunctions;

public class Sigmoid implements ActivationFunction {
    @Override
    public double function(double x) {
        return 1d / (1d + Math.exp(-x));
    }
    
    @Override
    public double derivative(double x) {
        if(Double.isInfinite(x)) return 0;
        return (Math.exp(-x) / Math.pow(1 + Math.exp(-x), 2));
    }
    
    @Override
    public double where(double y) {
        return - Math.log(1/y - 1);
    }
}
