package de.jensostertag.cnn.activationfunctions;

public class LayerActivation {
    public static Object activate(ActivationFunction activationFunction, Object input) {
        if(input instanceof Double input0d) {
            return activationFunction.function(input0d);
        } else if(input instanceof double[] input1d) {
            double[] output = new double[input1d.length];
            for(int i = 0; i < output.length; i++)
                output[i] = activationFunction.function(input1d[i]);
            return output;
        } else if(input instanceof double[][] input2d) {
            double[][] output = new double[input2d.length][];
            for(int i = 0; i < output.length; i++)
                output[i] = new double[input2d[i].length];
            
            for(int i = 0; i < output.length; i++)
                for(int j = 0; j < output[i].length; j++)
                    output[i][j] = activationFunction.function(input2d[i][j]);
            
            return output;
        } else if(input instanceof double[][][] input3d) {
            double[][][] output = new double[input3d.length][][];
            for(int i = 0; i < output.length; i++) {
                output[i] = new double[input3d[i].length][];
                for(int j = 0; j < output[i].length; j++)
                    output[i][j] = new double[input3d[i][j].length];
            }
    
            for(int i = 0; i < output.length; i++)
                for(int j = 0; j < output[i].length; j++)
                    for(int k = 0; k < output[i][j].length; k++)
                        output[i][j][k] = activationFunction.function(input3d[i][j][k]);
            
            return output;
        } else
            throw new IllegalArgumentException("This Method only allows 0-Dimensional up to 3-Dimensional Double Values as Input");
    }
}
