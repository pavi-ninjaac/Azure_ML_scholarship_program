# Azure_ML_scholarship_program
Lesson : 2 Introduction to Machine Learning

# What is Machine Learning?
  Machine learning is a data science technique used to extract patterns from data, allowing computers to identify related data, and forecast future outcomes, behaviors, and trends. <br/ >
> Gives computer the eligibility to learn without being explicitily programmed.

# Common types of data:

- 1) Numerical (continuous)
- 2) catogarical 
- 3) Text
- 4) Images
- 5) time-series
- 6) Tabular data -a very useful foramt of data that will make the prediction process and analysis process very easy.

# Scaling the data:
  Scaling data means transforming it so that the values fit within some range or scale, such as 0–100 or 0–1.<br/>
  two ways to scae a data: <br/>
  - 1) Standardization
  - 2) Normalization
## standardization:
  rescales data so that it has a mean of 0 and a standard deviation of 1.<br/>
> (𝑥 − 𝜇)/𝜎 here 𝜇 -is the mean and 𝜎 is standard deviation( the mean of the squerd distance from the mean)

## Normalization:
rescales the data into the range [0, 1]<br/>
> (𝑥 −𝑥𝑚𝑖𝑛)/(𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛) <br/> <br/>

> divide the each by the higest range . like for image data noramalization devide the each value by the 255 because the 255 is the higest range ( imgae pixel will value from 0 to 255)

# Encoding the categorical data:
  In ordinal encoding, we simply convert the categorical data into integer codes ranging from 0 to (number of categories – 1). Let's look again at our example table of clothing products:
SKU    |	Make |	Color |	Quantity |	Price|
|------|-------|--------|----------|-------|
|908721|Guess	 | Blue   |   	789  |	45.33|
|456552|Tillys |Red	    |244       |	22.91|
|789921|	A&F	 |Green	  |387	     |25.92  |

If we apply ordinal encoding to the Make property, we get the following:<br/>
|Make	 |Encoding |
|------|---------|
|A&F	 |0        |
|Guess |	1      |
|Tillys|	2      | 

### Drawback:
One of the potential drawbacks to this approach is that it implicitly assumes an order across the categories. In the above example, Blue (which is encoded with a value of 2) seems to be more than Red (which is encoded with a value of 1), even though this is in fact not a meaningful way of comparing those values. This is not necessarily a problem, but it is a reason to be cautious in terms of how the encoded data is used.

# One-hot encoding:
One-hot encoding is a very different approach. In one-hot encoding, we transform each categorical value into a column. If there are n categorical values, n new columns are added.<br/>
For example, the Color property has three categorical values: Red, Green, and Blue, so three new columns Red, Green, and Blue are added.

If an item belongs to a category, the column representing that category gets the value 1, and all other columns get the value 0. For example, item 908721 (first row in the table) has the color blue, so we put 1 into that Blue column for 908721 and 0 into the Red and Green columns. Item 456552 (second row in the table) has color red, so we put 1 into that Red column for 456552 and 0 into the Green and Blue columns.

|SKU	 |A&F |	Guess |	Tillys	| Red	 |Green |	Blue |	Quantity |	Price |
|------|----|-------|---------|------|------|------|-----------|--------|
|908721|	0	|1      |	0       | 	0	 |0     | 	1  | 	789      | 	45.33 | 
|456552|	0	| 0	    |1        | 	1  |	0   | 	0  |	244      |	22.91 |
|789921|	1	|0      |	0       |	0    | 	1   | 	0	 | 387       |	25.92 |

# Image data:
- 1) In grayscale images, each pixel can be represented by a single number, which typically ranges from 0 to 255. This value determines how dark the pixel appears (e.g., 0 is black, while 255 is bright white).
- 2) In colored images, each pixel can be represented by a vector of three numbers (each ranging from 0 to 255) for the three primary color channels: red, green, and blue. These three red, green, and blue (RGB) values are used together to decide the color of that pixel. For example, purple might be represented as 128, 0, 128 (a mix of moderately intense red and blue, with no green).
#### Depth:
The number of channels required to represent the color is known as the color depth or simply depth. With an RGB image, depth = 3, because there are three channels (Red, Green, and Blue). In contrast, a grayscale image has depth = 1, because there is only one channel.
#### Encoding the image:
There things are nedded for that
- 1) Horizontal position of each pixel
- 2) Vertical position of each pixel
- 3) Color of each pixel
Thus, we can fully encode an image numerically by using a vector with three dimensions. The size of the vector required for any given image would be the height * width * depth of that image.


# Text data:
#### Normalization
One of the challenges that can come up in text analysis is that there are often multiple forms that mean the same thing.<br/>
> Text normalization is the process of transforming a piece of text into a canonical (official) form.

#### Lemmatization
Lemmatization is an example of normalization. A lemma is the dictionary form of a word and lemmatization is the process of reducing multiple inflections to that single dictionary form. for example is,are,were are all converted to "be".

#### Tokanization
Here we have tokenized the text (i.e., split each string of text into a list of smaller parts or tokens), removed stop words (the), and standardized spelling (changing lazzy to lazy).
#### Vectorization
After we have normalized the text, we can take the next step of actually encoding it in a numerical form. The goal here is to identify the particular features of the text that will be relevant to us for the particular task we want to perform—and then get those features extracted in a numerical form that is accessible to the machine learning algorithm. Typically this is done by text vectorization—that is, by turning a piece of text into a vector. Remember, a vector is simply an array of numbers—so there are many different ways that we can vectorize a word or a sentence, depending on how we want to use it. Common approaches include:

- 1) Term Frequency-Inverse Document Frequency (TF-IDF) vectorization
- 2) Word embedding, as done with Word2vec or Global Vectors (GloVe)


## Model vs Algorithm
>> Models are the specific representations learned from data

>> Algorithms are the processes of learning

Model=Algorithm(Data)<br/>
##### Algorithms:
We can think of an algorithm as a mathematical tool that can usually be represented by an equation as well as implemented in code. For example, y = Wx + b is an algorithm that can be used to calculate y from x if the values for W and b are known. But how do we get W and b?
We can plug the data into the algorithm and calculate W = 1 and b = 0. We would say that that the algorithm was run on the data and learned the values ofr W and b. The output of the learning process is W = 1 and b = 0.
## models:
Machine learning models are outputs or specific representations of algorithms that run on data. A model represents what is learned by a machine learning algorithm on the data.

In the previous example, y = 1*x + 0 is the model we obtained from running the algorithm y = Wx + b on the training data. We can also say that y = 1*x + 0 is the model that can be used to predict y from x.

A machine learning model can also be written in a set of weights or coefficients instead of a full equation. Looking at the previous example, since we know the algorithm, it is redundant to keep the full equation y = 1*x + 0. All we need are the weights (or coefficients) W = 1 and b = 0. Thus, we can also think of a model as a set of weights (or coefficients) that have been learned.

## Learning function
As mentioned earlier, we can generally think of a machine learning algorithm as a process for learning, and models as specific representations that we train using data. In essence, machine learning algorithms aim to learn a target function (ff) that describes the mapping between data input variables (XX) and an output variable (YY).
>> Y=f(X)
Since the process extrapolates from a limited set of values, there will always be an error ee which is independent of the input data (XX) such that:
>> Y=f(X)+e


Note that the irreducible error we're discussing here is different from the model error we talked about earlier in the lesson. Irreducible error is caused by the data collection process—such as when we don't have enough data or don't have enough data features. In contrast, the model error measures how much the prediction made by the model is different from the true output. The model error is generated from the model and can be reduced during the model learning process.<br/>

### Parametric MOdel
Parametric machine learning algorithms make assumptions about the mapping function and have a fixed number of parameters. No matter how much data is used to learn the model, this will not change how many parameters the algorithm has. With a parametric algorithm, we are selecting the form of the function and then learning its coefficients using the training data.<br/>

An example of this would be the approach used in linear regression algorithms, where the simplified functional form can be something like:<br/>
>> B0+B1*X1+B2*X2=0
### Non-parametric algorithms
Non-parametric algorithms do not make assumptions regarding the form of the mapping function between input data and output. Consequently, they are free to learn any functional form from the training data.<br/>

A simple example is the K-nearest neighbors (KNN) algorithm, which we'll discuss in more detail later in the course. KNN does not make any assumptions about the functional form, but instead uses the pattern that points have similar output when they are close.





