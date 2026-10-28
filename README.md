# Not-a-digit
Not a Digit is a machine learning project developed in Python using PyTorch. It looks at how two neural networks recognise handwritten digits and how their predictions differ when the input is not a number.

One of the project features is the interactive drawing interface where users can draw something and see how both models interpret the same image.

# How It Works
The project uses two neural networks, which they call Brain A and Brain B. Each model is trained differently so that you can compare their predictions.

# Brain A
The model Brain A has been trained on the MNIST dataset. This dataset consists of handwritten digits for numbers ranging from 0-9. The model will always give an output between one of the 10 classes, meaning it will always give an answer for a digit even if it is a letter or any other shape.

# Brain B
Brain B was created using handwritten digits from the MNIST data set and handwritten letters from the EMNIST data set. This means that there are eleven classes to classify between, with numbers ranging from 0-9 and a further class for anything that isn’t a number.

Whilst training, any letters will be classified into the non number class. This means that Brain B can identify if an input may not be a number, however it doesn’t always predict correctly.

# The Drawing Interface
You can create your own images here and view how each model is predicting. As both networks are seeing the same image you can easily see the difference in the results caused by different training.

You can try drawing a number, a letter, or another shape and observe how each model responds.

# Neural Network Architecture
An input of 28 x 28 grayscale pixels is fed into the network, the image is flattened to give 784 inputs. These inputs go through a hidden layer of 128 nodes and then through a ReLU activation function. Finally, there is an output layer which gives scores for each class.

Brain A has 10 classes to output whilst Brain B has 11 classes.

# Training
Both networks were trained using Adam optimizer with the cross entropy loss function and learning rate of 0.001.

The MNIST data set was used to provide examples of handwritten digits. Brain B also used EMNIST letters data set to assign to the extra non digit class. Once the parameters of the model had been trained they were saved so that the application could load the models and make predictions.

# Image Processing
Before being input into either model, the user's drawing is pre-processed.

The application takes the drawing and puts it onto a white background. It then converts the image to grayscale, inverts the pixels and crops the image down to the drawing. Finally, the image is resized and centred within a 28x28 canvas. All pixel values are then normalised to be between 0 and 1.

This makes the models able to process the user's drawing in a way that is more similar to how they were trained.

# Results
I tested both brains on pictures they had never seen.

| | Brain A | Brain B |
|---|---|---|
| Digits read correctly | 96.8% | 97.0% |
| Real digits wrongly thrown out | 10.7% | 0.4% |
| Letters wrongly let through | 46.8% | 0.6% |
| Clothes wrongly let through | 73.7% | 7.0% |

A lower percentage means better results, except in the first row. "Let through" which means the brain wrongly let thought a letter or a piece of clothing as a digit.

# Built With
. Python

. PyTorch

. Torchvision

. NumPy

. Pillow

. Gradio

# Running the Project
Install the required libraries:
```python
pip install torch torchvision gradio numpy pillow
```

Start the application by running:
```python
python Not-a-digit-v1.py
```
# Limitations
As Brain B was trained on handwritten letters it cannot be assumed to understand all shapes/symbols as being non digits. Both models will have errors when they are presented with unusual handwriting/graphics.

Overall, the aim of this project was to examine how different methods of classifying images of handwritten data can be used and how the training data can affect a model's predictions.

