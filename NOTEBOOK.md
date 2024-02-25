# Intro
I should probably learn how to use a proper notebook like Jupyter for this.

# Notes

## Architecture
win_and_angle_detector was the final architecture I got to work. Some issues I encountered:

1. Getting the output dimensions to work for 400x400 images required adding an extra pooling layer.
2. Numeric instability in the sigmoid function was throwning NaNs. I 'fixed' this by going from 16 to 32 bit floats which of course doubles the model size (it's still pretty small at about 40MB of parameters). I haven't looked into mixed precision models...maybe only running the sigmoid at 32 bits would work.
3. Thanks to my low volume of training data, training had a luck component to it and seemed to do well or not based on how the weights got intialized. Maybe I could fix this with more training data, particularly on the classification portion since all the images are of locks, there are no random images to add a nice gradient to the "I am not a successful lockpick" class, which includes not just unsuccessful lockpicks, but all images that are of anything other than a lock including noise.
4. I was able to 'luck out' and get a good gradient hole for the quality (angle) metric but it would not converge on the classification. To attempt to remedy this, I loaded the train with the good quality loss, re-initialized just the classification layers, and retrained. I got good losses for both and was happy.
5. Oh yeah the unlock was not trying to reuse the same Linear layer on top of the CNN for both classification and angle. I duplicated it.
