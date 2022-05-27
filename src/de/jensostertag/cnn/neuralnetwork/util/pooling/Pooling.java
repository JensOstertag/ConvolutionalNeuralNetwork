package de.jensostertag.cnn.neuralnetwork.util.pooling;

public class Pooling {
    public final PoolingType POOLING_TYPE;
    public final int POOLING_SIZE;
    
    public Pooling(PoolingType POOLING_TYPE, int POOLING_SIZE) {
        this.POOLING_TYPE = POOLING_TYPE;
        if(POOLING_SIZE > 0)
            this.POOLING_SIZE = POOLING_SIZE;
        else
            throw new IllegalArgumentException("PoolingSize is supposed to be a positive Integer");
    }
}
