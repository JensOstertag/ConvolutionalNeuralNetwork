package de.jensostertag.cnn.neuralnetwork;

import de.jensostertag.cnn.neuralnetwork.layers.Layer;

import java.util.ArrayList;

public class CNN {
    protected ArrayList<Layer> layers = new ArrayList<>();
    
    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }
}
