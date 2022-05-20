package de.jensostertag.cnn.activationfunctions;

public class Relu implements ActivationFunction {
    @Override
    public double function(double x) {
        return Math.max(0, x);
    }
    
    @Override
    public double derivative(double x) {
        if(x < 0) return 0;
        return 1;
    }
    
    @Override
    public double where(double y) {
        return Math.max(0, y);
    }
}
