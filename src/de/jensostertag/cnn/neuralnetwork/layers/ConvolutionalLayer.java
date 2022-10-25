package de.jensostertag.cnn.neuralnetwork.layers;

import de.jensostertag.cnn.activationfunctions.ActivationFunction;
import de.jensostertag.cnn.activationfunctions.LayerActivation;
import de.jensostertag.cnn.neuralnetwork.Config;
import de.jensostertag.cnn.neuralnetwork.util.Matrices;
import de.jensostertag.cnn.neuralnetwork.util.padding.Padding;
import de.jensostertag.cnn.neuralnetwork.util.padding.PaddingType;

import java.util.Arrays;

public class ConvolutionalLayer implements Layer {
    private final int INPUT_CHANNELS;
    private final int OUTPUT_CHANNELS;
    private final int INPUT_HEIGHT;
    private final int INPUT_WIDTH;
    private final int KERNEL_SIZE;
    private final Padding PADDING;
    public double[][][][] kernel;
    public double[][][] biases;
    private final ActivationFunction activationFunction;
    
    private Object input;
    
    public ConvolutionalLayer(int INPUT_CHANNELS, int OUTPUT_CHANNELS, int INPUT_HEIGHT, int INPUT_WIDTH, int KERNEL_SIZE, PaddingType paddingType, ActivationFunction activationFunction) {
        this.INPUT_CHANNELS = INPUT_CHANNELS;
        this.OUTPUT_CHANNELS = OUTPUT_CHANNELS;
        this.INPUT_HEIGHT = INPUT_HEIGHT;
        this.INPUT_WIDTH = INPUT_WIDTH;
        this.KERNEL_SIZE = KERNEL_SIZE;
        this.PADDING = new Padding(paddingType, this.INPUT_HEIGHT, this.INPUT_WIDTH);
        this.kernel = new double[this.OUTPUT_CHANNELS][this.INPUT_CHANNELS][this.KERNEL_SIZE][this.KERNEL_SIZE];
        int ps = this.PADDING.getEffectivePaddingSize(this.KERNEL_SIZE);
        this.biases = new double[this.OUTPUT_CHANNELS][this.INPUT_HEIGHT - 2 * ps][this.INPUT_WIDTH - 2 * ps];
        for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
            for(int j = 0; j < this.INPUT_CHANNELS; j++)
                this.kernel[i][j] = Matrices.randomMatrix(this.KERNEL_SIZE, this.KERNEL_SIZE, Config.DEFAULT_WEIGHT_MIN, Config.DEFAULT_WEIGHT_MAX);
            this.biases[i] = Matrices.randomMatrix(this.INPUT_HEIGHT - 2 * ps, this.INPUT_WIDTH - 2 * ps, Config.DEFAULT_WEIGHT_MIN, Config.DEFAULT_WEIGHT_MAX);
        }
        this.activationFunction = activationFunction;
    }
    
    @Override
    public double[][][] propagate(Object input) {
        if(input instanceof double[][][] layerInput) {
            this.input = input;
            
            if(Matrices.validateSize(layerInput, this.INPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH)) {
                double[][][] output = new double[this.OUTPUT_CHANNELS][][];
                
                for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                    output[i] = this.biases[i];
                    for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                        double[][] inputMatrix = this.PADDING.applyPadding(layerInput[j], this.KERNEL_SIZE);
                        double[][] convolved = Matrices.convolve(inputMatrix, this.kernel[i][j]);
                        output[i] = Matrices.add(output[i], convolved);
                    }
                }
                
                return (double[][][]) LayerActivation.activate(this.activationFunction, output);
            } else
                throw new IllegalArgumentException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a 3-Dimensional Double Array");
    }
    
    @Override
    public Object backPropagate(Object d_L_d_Y, double learningRate) {
        if(d_L_d_Y instanceof double[][][] gradient && this.input instanceof double[][][] layerInput) {
            int eps = this.PADDING.getEffectivePaddingSize(this.KERNEL_SIZE);
            if(Matrices.validateSize(gradient, this.OUTPUT_CHANNELS, this.INPUT_HEIGHT - 2 * eps, this.INPUT_WIDTH - 2 * eps)) {
                if(Matrices.validateSize(layerInput, this.INPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH)) {
                    double[][][][] d_L_d_K = new double[this.OUTPUT_CHANNELS][this.INPUT_CHANNELS][this.KERNEL_SIZE][this.KERNEL_SIZE];
                    double[][][] d_L_d_B = new double[this.OUTPUT_CHANNELS][this.INPUT_HEIGHT - 2 * eps][this.INPUT_WIDTH - 2 * eps];
                    double[][][] d_L_d_X = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
    
                    double[][][] net = new double[this.OUTPUT_CHANNELS][][];
    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        net[i] = this.biases[i];
                        for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                            double[][] inputMatrix = this.PADDING.applyPadding(layerInput[j], this.KERNEL_SIZE);
                            double[][] convolved = Matrices.convolve(inputMatrix, this.kernel[i][j]);
                            net[i] = Matrices.add(net[i], convolved);
                        }
                    }
                    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        double[][] d_Y_d_net = (double[][]) LayerActivation.derive(this.activationFunction, net[i]);
                        double[][] workingGradient = Matrices.dotProduct(gradient[i], d_Y_d_net);
                        
                        for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                            double[][] inputMatrix = this.PADDING.applyPadding(layerInput[j], this.KERNEL_SIZE);
                            d_L_d_K[i][j] = Matrices.convolve(inputMatrix, workingGradient);
    
                            int ps = this.PADDING.getPaddingSize(this.KERNEL_SIZE);
                            double[][] padded = Matrices.center(workingGradient, this.INPUT_HEIGHT + 2*ps, this.INPUT_WIDTH + 2*ps);
                            d_L_d_X[j] = Matrices.add(d_L_d_X[j], Matrices.convolve(padded, Matrices.rotate(this.kernel[i][j])));
                        }
                        d_L_d_B[i] = gradient[i];
                    }
                    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        for(int j = 0; j < this.INPUT_CHANNELS; j++)
                            this.kernel[i][j] = Matrices.add(this.kernel[i][j], Matrices.multiplyConstant(d_L_d_K[i][j], -learningRate));
                        this.biases[i] = Matrices.add(this.biases[i], Matrices.multiplyConstant(d_L_d_B[i], -learningRate));
                    }
                    
                    return d_L_d_X;
                } else
                    throw new IllegalArgumentException("Input is not of correct Size");
            } else
                throw new IllegalArgumentException("Gradient is not of correct Size");
        } else
            throw new IllegalArgumentException("Gradient and Input are supposed to be 3-Dimensional Double Arrays");
    }

    @Override
    public ConvolutionalLayer copy() {
        double[][][][] kernel = new double[this.kernel.length][][][];
        double[][][] biases = new double[this.biases.length][][];

        for(int i = 0; i < kernel.length; i++) {
            kernel[i] = new double[this.kernel[i].length][][];
            for(int j = 0; j < kernel[i].length; j++) {
                kernel[i][j] = new double[this.kernel[i][j].length][];
                for(int k = 0; k < kernel[i][j].length; k++) {
                    System.arraycopy(this.kernel[i][j][k], 0, kernel[i][j][k], 0, this.kernel[i][j][k].length);
                }
            }
        }

        for(int i = 0; i < biases.length; i++) {
            biases[i] = new double[this.biases[i].length][];
            for(int j = 0; j < biases[i].length; j++) {
                System.arraycopy(this.biases[i][j], 0, biases[i][j], 0, this.biases[i][j].length);
            }
        }

        ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(this.INPUT_CHANNELS, this.OUTPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH, this.KERNEL_SIZE, this.PADDING.PADDING_TYPE, this.activationFunction);
        convolutionalLayer.kernel = kernel;
        convolutionalLayer.biases = biases;

        return convolutionalLayer;
    }
}
