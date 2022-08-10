package de.jensostertag.cnn.activationfunctions;

public class Linear implements ActivationFunction {
    @Override
    public double function(double x) {
        return x;
    }
    
    @Override
    public double derivative(double x) {
        return 1;
    }
    
    @Override
    public double where(double y) {
        return y;
    }
}
