package de.jensostertag.cnn.activationfunctions;

public class Step implements ActivationFunction {
    @Override
    public double function(double x) {
        return Math.max(0, Math.min(1, x));
    }
    
    @Override
    public double derivative(double x) {
        if(x < 0 || x > 1) return 0;
        return 1;
    }
    
    @Override
    public double where(double y) {
        return Math.max(0, Math.min(1, y));
    }
}
