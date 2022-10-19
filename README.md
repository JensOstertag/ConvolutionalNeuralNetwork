# Convolutional Neural Network
This Repository contains my own Convolutional Neural Network usable for Object Dection within Images or to 
decide over even simpler Tasks.

For the Sake of Simplicity, I've developed the CNN using pure and Object-Oriented Java.

## How to use

### The Config
At First, let's have a Look at the Config.

The first Constants are used after each Training Epoch, when the Learning Rate will be changed.
- ``STARTING_LEARNING_RATE``: Initial Learning Rate
- ``ADJUST_LEARNING_RATE_DIFF`` and ``ADJUST_LEARNING_RATE``: At the Start of an Epoch, the Accuracy Difference of the previous
and the Epoch before will be compared. If the absolute Difference is smaller or equal to ``ADJUST_LEARNING_RATE_DIFF``, the Learning Rate
for this Epoch will be multiplied by (1 + ``ADJUST_LEARNING_RATE``). Else, it will be divided by (1 + ``ADJUST_LEARNING_RATE``).
- ``LEARNING_RATE_MIN`` and ``LEARNING_RATE_MAX``: To prevent the Learning Rate from getting too small or large, these Constants define the
lower and upper Limit of the Learning Rate.
- ``INERTIA``: To prevent the Learning Rate from changing too much each Epoch, the Inertia is used to take a part of the previous Learning
Rate to the next Epoch. The Parts will be averaged out. Not used by now.

Next, the initial Weights are set with
- ``DEFAULT_WEIGHT_MIN`` and ``DEFAULT_WEIGHT_MAX``: Every single Weight will start with a Value between both of the Constants.

You can also set Training Limits:
- ``AMING_ACCURACY``: When the Model reached an Accuracy higher than this Value during Training, the Training Process will stop and be
considered as a Success.
- ``MAX_TRAINING_ITERATIONS``: When the Model trained with more Epochs than this Value, the Training Process will be aborted. The Amount of
Epochs that are necessary for a successful Training will vary, depending on the Problem Complexity and Training Dataset Length.

The last Section is used for Debugging Settings.
- ``LOGGING_LEVEL``: Select how detailed the Debugging Messages will be.

### Instantiating a Network

### Adding Layers

### Training

### Usage