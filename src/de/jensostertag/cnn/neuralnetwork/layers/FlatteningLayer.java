package de.jensostertag.cnn.neuralnetwork.layers;

public class FlatteningLayer implements Layer {
    private final int INPUT_CHANNELS;
    private final int INPUT_WIDTH;
    private final int INPUT_HEIGHT;
    
    public FlatteningLayer(int INPUT_CHANNELS, int INPUT_WIDTH, int INPUT_HEIGHT) {
        this.INPUT_CHANNELS = INPUT_CHANNELS;
        this.INPUT_WIDTH = INPUT_WIDTH;
        this.INPUT_HEIGHT = INPUT_HEIGHT;
    }
    
    @Override
    public Object propagate(Object input) {
        if(input instanceof double[][][] layerInput) {
            if(layerInput.length == this.INPUT_CHANNELS) {
                double[] output = new double[this.INPUT_CHANNELS * this.INPUT_WIDTH * this.INPUT_HEIGHT];
                
                for(int i = 0; i < layerInput.length; i++) {
                    if(layerInput[i].length == this.INPUT_HEIGHT) {
                        for(int j = 0; j < layerInput[i].length; j++) {
                            if(layerInput[i][j].length == this.INPUT_WIDTH) {
                                for(int k = 0; k < layerInput[i][j].length; k++) {
                                    int index = i * this.INPUT_HEIGHT * this.INPUT_WIDTH + j * this.INPUT_WIDTH + k;
                                    output[index] = layerInput[i][j][k];
                                }
                            } else
                                throw new IllegalStateException("Input is not of correct Size");
                        }
                    } else
                        throw new IllegalStateException("Input is not of correct Size");
                }
                
                return output;
            } else
                throw new IllegalStateException("Input is not of correct Size");
        } else
            throw new IllegalArgumentException("Input is supposed to be a 3-Dimensional Double Array");
    }
    
    @Override
    public Object mistakes(Object previousMistakes, Object layerOutput) {
        if(previousMistakes instanceof double[] mistakes && layerOutput instanceof double[] output) {
            double[][][] newMistakes = new double[this.INPUT_CHANNELS][this.INPUT_HEIGHT][this.INPUT_WIDTH];
            
            int valuesPerMatrix = this.INPUT_HEIGHT * this.INPUT_WIDTH;
            
            for(int i = 0; i < mistakes.length; i++) {
                double mistake = mistakes[i];
                int index = i;
                int channel = index / valuesPerMatrix;
                index -= channel * valuesPerMatrix;
                int row = index / this.INPUT_WIDTH;
                index -= row * this.INPUT_WIDTH;
                int col = index;
                
                newMistakes[channel][row][col] = mistake;
            }
            
            return newMistakes;
        } else
            throw new IllegalArgumentException("PreviousMistakes and LayerOutput are supposed to be Double Array");
    }
    
    @Override
    public void optimizeWeights(Object previousMistakes, Object layerOutput, double learningRate) {}
}
