package de.jensostertag.cnn.dataset;

public class Dataset {
    private Object[] inputs = new Object[0];
    private Object[] expectedOutputs = new Object[0];
    private DatasetState state = DatasetState.INVALID;
    
    /**
     * Validate the Dataset
     * @return Validation successful
     */
    public boolean validateDataset() {
        if(this.inputs.length != this.expectedOutputs.length || this.inputs.length == 0)
            return false;
    
        for(int i = 0; i < this.inputs.length; i++)
            if(this.inputs[i].getClass() != this.inputs[0].getClass()
                    || this.expectedOutputs[i].getClass() !=  this.expectedOutputs[0].getClass())
                return false;
        
        this.state = DatasetState.VALIDATED;
        return true;
    }
    
    /**
     * Insert data to the Dataset
     * @param input Input Data
     * @param expectedOutput Expected Output Data
     */
    public void insertData(Object input, Object expectedOutput) {
        if(this.state == DatasetState.INVALID) {
            Object[] newInputs = new Object[this.inputs.length + 1];
            Object[] newExpectedOutputs = new Object[this.expectedOutputs.length + 1];
            
            for(int i = 0; i < this.inputs.length; i++)
                newInputs[i] = this.inputs[i];
            for(int i = 0; i < this.expectedOutputs.length; i++)
                newExpectedOutputs[i] = this.expectedOutputs[i];
            newInputs[newInputs.length - 1] = input;
            newExpectedOutputs[newExpectedOutputs.length - 1] = expectedOutput;
            
            this.inputs = newInputs;
            this.expectedOutputs = newExpectedOutputs;
        } else
            throw new IllegalStateException("Cannot insert Data to a validated Dataset");
    }
    
    /**
     * Check whether a Dataset has been validated or not
     * @return Dataset is valid
     */
    public boolean isValid() {
        return this.state == DatasetState.VALIDATED;
    }
    
    /**
     * Get the Input at an Index
     * @param index Index
     * @return Input at given Index
     */
    public Object getInput(int index) {
        if(isValid())
            if(index >= 0 && index < this.inputs.length)
                return this.inputs[index];
            else
                throw new IndexOutOfBoundsException("Index " + index + " out of Bounds for Array Length " + this.inputs.length);
        else
            throw new IllegalStateException("Dataset must be validated");
    }
    
    /**
     * Get the expected Output at an Index
     * @param index Index
     * @return Expected Output at given Index
     */
    public Object getExpectedOutput(int index) {
        if(isValid())
            if(index >= 0 && index < this.expectedOutputs.length)
                return this.expectedOutputs[index];
            else
                throw new IndexOutOfBoundsException("Index " + index + " out of Bounds for Array Length " + this.expectedOutputs.length);
        else
            throw new IllegalStateException("Dataset must be validated");
    }
    
    /**
     * Get an Array of all Inputs
     * @return Array of all Inputs
     */
    public Object[] getInputs() {
        return this.inputs;
    }
    
    /**
     * Get an Array of all expected Outputs
     * @return Array of all expected Outputs
     */
    public Object[] getExpectedOutputs() {
        return this.expectedOutputs;
    }
}
