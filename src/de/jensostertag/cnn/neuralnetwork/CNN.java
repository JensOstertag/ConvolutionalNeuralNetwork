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
}
