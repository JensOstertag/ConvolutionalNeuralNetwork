package de.jensostertag.cnn.neuralnetwork;

import java.util.logging.Level;

public class Config {
    public static final double STARTING_LEARNING_RATE = .05;
    public static final double ADJUST_LEARNING_RATE_DIFF = .01;
    public static final double ADJUST_LEARNING_RATE = .1;
    public static final double LEARNING_RATE_MIN = .0025;
    public static final double LEARNING_RATE_MAX = .5;
    
    public static final double INERTIA = .05;
    
    public static final double DEFAULT_WEIGHT_MIN = -.3;
    public static final double DEFAULT_WEIGHT_MAX = .3;
    
    public static final double AIMING_ACCURACY = .95;
    public static final int MAX_TRAINING_ITERATIONS = 100;
    
    public static final Level LOGGING_LEVEL = Level.FINE;
}
