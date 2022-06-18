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
    public Object backPropagate(Object d_L_d_Y, Object input, double learningRate) {
        if(d_L_d_Y instanceof double[][][] gradient && input instanceof double[][][] layerInput) {
            int eps = this.PADDING.getEffectivePaddingSize(this.KERNEL_SIZE);
            if(Matrices.validateSize(gradient, this.OUTPUT_CHANNELS, this.INPUT_HEIGHT - 2 * eps, this.INPUT_WIDTH - 2 * eps)) {
                if(Matrices.validateSize(layerInput, this.INPUT_CHANNELS, this.INPUT_HEIGHT, this.INPUT_WIDTH)) {
                    double[][][][] d_L_d_K = new double[this.OUTPUT_CHANNELS][this.INPUT_CHANNELS][this.KERNEL_SIZE][this.KERNEL_SIZE];
                    double[][][] d_L_d_B = new double[this.OUTPUT_CHANNELS][this.INPUT_HEIGHT - 2 * eps][this.INPUT_WIDTH - 2 * eps];
                    double[][][] d_L_d_X = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
                    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        //double[][] derivedGradient = (double[][]) LayerActivation.derive(this.activationFunction, gradient[i]);
                        //double[][] workingGradient = Matrices.dotProduct(gradient[i], derivedGradient);
                        double[][] workingGradient = gradient[i];
                        
                        for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                            double[][] inputMatrix = this.PADDING.applyPadding(layerInput[j], this.KERNEL_SIZE);
                            d_L_d_K[i][j] = Matrices.convolve(inputMatrix, workingGradient);
                        }
                        d_L_d_B[i] = gradient[i];
                    }
                    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        for(int j = 0; j < this.INPUT_CHANNELS; j++)
                            this.kernel[i][j] = Matrices.add(this.kernel[i][j], Matrices.multiplyConstant(d_L_d_K[i][j], -learningRate));
                        this.biases[i] = Matrices.add(this.biases[i], Matrices.multiplyConstant(d_L_d_B[i], -learningRate));
                    }
                    
                    return null;
                    
                    /*double[][][][] d_L_d_K = new double[this.OUTPUT_CHANNELS][this.INPUT_CHANNELS][this.KERNEL_SIZE][this.KERNEL_SIZE];
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        double[][] derivedGradient = (double[][]) LayerActivation.derive(this.activationFunction, gradient[i]);
                        double[][] workingGradient = Matrices.dotProduct(gradient[i], derivedGradient);
                        
                        for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                            int ps = this.PADDING.getPaddingSize(this.KERNEL_SIZE);
                            for(int k = 0; k < layerInput[j].length - 2*ps; k++) {
                                for(int l = 0; l < layerInput[j][k].length - 2*ps; l++) {
                                    double[][] subMatrix = Matrices.subMatrix(layerInput[j], k, l, this.KERNEL_SIZE, this.KERNEL_SIZE);
                                    d_L_d_K[i][j] = Matrices.add(d_L_d_K[i][j], Matrices.multiplyConstant(subMatrix, workingGradient[k][l]));
                                }
                            }
                        }
                    }
                    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                            double[] addKernel = new double[this.KERNEL_SIZE * this.KERNEL_SIZE + 1];
                            for(int k = 0; k < addKernel.length; k++)
                                if(k < this.KERNEL_SIZE * this.KERNEL_SIZE)
                                    addKernel[k] = d_L_d_K[i][j][k / this.KERNEL_SIZE][k % this.KERNEL_SIZE];
                            this.kernel[i][j] = Matrices.add(this.kernel[i][j], Matrices.multiplyConstant(addKernel, -learningRate));
                        }
                    }
                    
                    return null;*/
                    
                    
                    
                    /*
                    double[][][] d_L_d_X = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
                    
                    for(int i = 0; i < this.OUTPUT_CHANNELS; i++) {
                        double[][] derivedGradient = (double[][]) LayerActivation.derive(this.activationFunction, gradient[i]);
                        double[][] workingGradient = Matrices.dotProduct(gradient[i], derivedGradient);
                        
                        for(int j = 0; j < this.INPUT_CHANNELS; j++) {
                            double[][] workingInput = this.PADDING.applyPadding(layerInput[j], this.KERNEL_SIZE);
                            double[][] d_L_d_K = Matrices.convolve(workingInput, workingGradient);
                            double d_L_d_B = Matrices.sum(workingGradient);
                            
                            double[][] currentKernel = new double[this.KERNEL_SIZE][this.KERNEL_SIZE];
                            for(int k = 0; k < currentKernel.length; k++)
                                for(int l = 0; l < currentKernel[k].length; l++)
                                    currentKernel[k][l] = this.kernel[i][j][k * currentKernel[k].length + l];
                            double[][] newKernel = Matrices.add(currentKernel, Matrices.multiplyConstant(d_L_d_K, -learningRate));
                            //double currentBias = this.kernel[i][j][this.kernel[i][j].length - 1];
                            double currentBias = 0;
                            double newBias = currentBias + d_L_d_B * learningRate;
                            double[] newKernelVector = new double[this.KERNEL_SIZE * this.KERNEL_SIZE + 1];
                            for(int k = 0; k < newKernelVector.length; k++) {
                                if(k < this.KERNEL_SIZE * this.KERNEL_SIZE)
                                    newKernelVector[k] = newKernel[k / this.KERNEL_SIZE][k % this.KERNEL_SIZE];
                                else
                                    newKernelVector[k] = newBias;
                            }
                            
                            int ps = this.PADDING.getPaddingSize(this.KERNEL_SIZE);
                            double[][] paddedGradient = Matrices.center(workingGradient, this.INPUT_HEIGHT + 2*ps, this.INPUT_WIDTH + 2*ps);
                            d_L_d_X[j] = Matrices.add(d_L_d_X[j], Matrices.convolve(paddedGradient, Matrices.rotate(currentKernel)));
                            
                            this.kernel[i][j] = newKernelVector;
                        }
                    }
                    
                    for(int i = 0; i < d_L_d_X.length; i++)
                        d_L_d_X[i] = Matrices.multiplyConstant(d_L_d_X[i], this.OUTPUT_CHANNELS);
                    
                    return d_L_d_X;
                    */
                } else
                    throw new IllegalArgumentException("Input is not of correct Size");
            } else
                throw new IllegalArgumentException("Gradient is not of correct Size");
        } else
            throw new IllegalArgumentException("Gradient and Input are supposed to be 3-Dimensional Double Arrays");
    }
}
