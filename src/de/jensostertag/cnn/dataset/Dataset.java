package de.jensostertag.cnn.dataset;

public class Dataset {
    private Object[] inputs = new Object[0];
    private Object[] expectedOutputs = new Object[0];
    private DatasetState state = DatasetState.INVALID;
    
    public boolean validateDataset() {
        if(this.inputs.length != this.expectedOutputs.length || this.inputs.length == 0)
            return false;
    
        for(int i = 0; i < this.inputs.length; i++)
            if(this.inputs[i].getClass() == this.inputs[0].getClass()
                    || this.expectedOutputs[i].getClass() ==  this.expectedOutputs[0].getClass())
                return false;
        
        this.state = DatasetState.VALIDATED;
        return true;
    }
    
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
    
    public boolean isValid() {
        return this.state == DatasetState.VALIDATED;
    }
    
    public Object getInput(int index) {
        if(isValid())
            if(index >= 0 && index < this.inputs.length)
                return this.inputs[index];
            else
                throw new IndexOutOfBoundsException("Index " + index + " out of Bounds for Array Length " + this.inputs.length);
        else
            throw new IllegalStateException("Dataset must be validated");
    }
    
    public Object getExpectedOutput(int index) {
        if(isValid())
            if(index >= 0 && index < this.expectedOutputs.length)
                return this.inputs[index];
            else
                throw new IndexOutOfBoundsException("Index " + index + " out of Bounds for Array Length " + this.inputs.length);
        else
            throw new IllegalStateException("Dataset must be validated");
    }
    
    public Object[] getInputs() {
        return this.inputs;
    }
    
    public Object[] getExpectedOutputs() {
        return this.expectedOutputs;
    }
}
