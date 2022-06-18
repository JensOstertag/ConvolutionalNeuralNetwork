package de.jensostertag.cnn.neuralnetwork.util.padding;

import java.util.Arrays;

public class Padding {
    public final PaddingType PADDING_TYPE;
    private final int INPUT_HEIGHT;
    private final int INPUT_WIDTH;
    
    public Padding(PaddingType PADDING_TYPE, int INPUT_HEIGHT, int INPUT_WIDTH) {
        this.PADDING_TYPE = PADDING_TYPE;
        this.INPUT_HEIGHT = INPUT_HEIGHT;
        this.INPUT_WIDTH = INPUT_WIDTH;
    }
    
    public int getPaddingSize(int kernelSize) {
        if(kernelSize % 2 == 1)
            return (kernelSize - 1) / 2;
        else
            throw new IllegalArgumentException("Invalid Kernel Size");
    }
    
    public int getEffectivePaddingSize(int kernelSize) {
        if(kernelSize % 2 == 1)
            return switch(this.PADDING_TYPE) {
                case SAME -> 0;
                case VALID -> (kernelSize - 1) / 2;
            };
        else
            throw new IllegalArgumentException("Invalid Kernel Size");
    }
    
    public double[][] applyPadding(double[][] input, int kernelSize) {
        if(kernelSize % 2 == 1) {
            if(input.length == this.INPUT_HEIGHT) {
                for(double[] inputVector : input)
                    if(inputVector.length != this.INPUT_WIDTH)
                        throw new IllegalArgumentException("Input is not of correct Size");
        
                if(this.PADDING_TYPE == PaddingType.SAME) {
                    int necessaryPadding = getPaddingSize(kernelSize);
                    double[][] output = new double[this.INPUT_WIDTH + 2 * necessaryPadding][this.INPUT_WIDTH + 2 * necessaryPadding];
                    
                    for(int i = 0; i < output.length; i++) {
                        if(i < necessaryPadding || i >= this.INPUT_HEIGHT + necessaryPadding)
                            Arrays.fill(output[i], 0);
                        else {
                            for(int j = 0; j < output[i].length; j++) {
                                if(j < necessaryPadding || j >= this.INPUT_WIDTH + necessaryPadding)
                                    output[i][j] = 0;
                                else
                                    output[i][j] = input[i - necessaryPadding][j - necessaryPadding];
                            }
                        }
                    }
                    
                    return output;
                } else if(this.PADDING_TYPE == PaddingType.VALID) {
                    return input;
                }
            } else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Invalid Kernel Size");
        
        return null;
    }
}
