package de.jensostertag.cnn.neuralnetwork;

import de.jensostertag.cnn.neuralnetwork.layers.FullyConnectedLayer;
import de.jensostertag.cnn.neuralnetwork.layers.Layer;

import java.util.ArrayList;

public class CNN {
    protected ArrayList<Layer> layers = new ArrayList<>();
    
    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }
    
    public Object[] propagate(Object input) {
        Object[] output = new Object[this.layers.size() + 1];
        output[0] = input;
    
        for(int i = 0; i < this.layers.size(); i++)
            output[i + 1] = this.layers.get(i).propagate(output[i]);
    
        return output;
    }
    
    public Object[] mistakes(Object[] output, double[] expectedOutput) {
        Object[] mistakes = new Object[this.layers.size()];
        
        for(int i = mistakes.length - 1; i >= 0; i--) {
            if(i == mistakes.length - 1) {
                if(this.layers.get(i) instanceof FullyConnectedLayer layer)
                    mistakes[i] = layer.outputMistakes(expectedOutput, output[i + 1]);
                else
                    throw new IllegalArgumentException("The last Layer in the Neural Network is supposed to be a Fully Connected Layer");
            } else {
                mistakes[i] = this.layers.get(i + 1).mistakes(mistakes[i + 1], output[i + 2]);
            }
        }
        
        return mistakes;
    }
}
