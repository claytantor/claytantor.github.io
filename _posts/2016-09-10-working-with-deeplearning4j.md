---
layout: post
title:  "A newbie uses deeplearning4j"
date:   2016-09-10 14:42:30 -0700
categories: deeplearning java
---

I admit it. I am ignorant. I also am not a wizard, information doesn't just get injected into my brain as if I was neo from the matrix. **I don't know kungfu**.
So in my mind its normal to ask dumb questions or misunderstand when coming to something new and fresh. Helping people be successful means being compassionate when people come to you and ask for help. That's where I am coming from with this post in my first impressions and issues I am seeing around using **deeplearning4j**.

Most enterprise companies understand this and spend a ton of time, money and energy on building documentation, cookbooks, how-tos and then nuturing a community that make understanding the essentials of how a technology is used easy.

The commercial company behind **deeplearning4j** is [skymind](https://skymind.io/). It positions itself as "Deep Learning for Enterprise Level Applications", which is an important claim because enterprise is not focused on science, or mathematics, its focused on business impact. Business concerns are different than academic ones and as a set of technologies like this technology's transition from primarily academic settings to commercial ones the product owner should spend time and have concern for business oriented users.

This post is about my experiences in the first week of using deeplearning4j as a business oriented user, the challenges I faced and my experiences.

## Important Links

* [All deeplearning4j Repos](https://github.com/deeplearning4j) - This is all the repos related to deeplearning4j, including the examples.
* [ND4j Repo On Github](https://github.com/deeplearning4j/nd4j) - ND4J is the mathematics foundation classes used by deeplearning4j.  
* [Andy Gibson's Glitter](https://gitter.im/agibsonccc) - *Newbie's beware.* Ask the wrong question to the group and Andy will start a DM with you and tell you that you are wasting his time. I found that AlexDBlack was helpful even though I asked him one very dumb question.

# What Slowed Me Down TL;DR

When trying to use **deeplearning4j** I ran into some specific challenges that slowed me down.

* Using INDArray  
* Classification and Correlation
* Labeling
* "End To End" Documentation
* Lack of Simple Recommendations for GPU

# Using INDArray
INDArray is a buffer, and its really great at being flexible for many different use cases because it can be used in N dimensional problems. All you need to do is use the shape to then split the pages up.

https://www.mathworks.com/help/matlab/math/multidimensional-arrays.html?requestedDomain=www.mathworks.com&requestedDomain=www.mathworks.com

![Buffer Pagination](https://www.mathworks.com/help/matlab/math/array_storage.gif)

The data buffer in this example is a single array.

`[0.3,2.4,3.6,1.3,3.4,1.6,2.3,0.4,0.6,3.3,1.4,0.6]`

and the shape: `[4,3]`

So when you look at the base implementation (BaseNDArray) of INDArray the toString method properly creates a model with the right N level dimension,
paged in a way that the shape tells you it should be. ie. A two dimensional shape would make a string representation:

```
[
   [0.3,2.4,3.6],
   [1.3,3.4,1.6],
   [2.3,0.4,0.6],
   [3.3,1.4,0.6]
]
```

and the BaseNDArray toString() method:

    {% highlight java %}
    /**
     * Generate string representation of the matrix.
     */
    @Override
    public String toString() {
        return new NDArrayStrings().format(this);
    }
    {% endhighlight %}

So when they implemented NDArrayStrings this is what they did. The to string methods all build json equivalent versions for the array:

    {% highlight java %}
    private String format(INDArray arr,int rank, int offset) {
        StringBuilder sb = new StringBuilder();
        if(arr.isScalar()) {
            if(arr instanceof IComplexNDArray)
                return ((IComplexNDArray) arr).getComplex(0).toString();
            return decimalFormat.format(arr.getDouble(0));
        }
        else if(rank <= 0)
            return "";

        else if(arr.isVector()) {
            sb.append("[");
            for(int i = 0; i < arr.length(); i++) {
                if(arr instanceof IComplexNDArray)
                    sb.append(((IComplexNDArray) arr).getComplex(i).toString());
                else
                    sb.append(String.format("%1$"+padding+"s",decimalFormat.format(arr.getDouble(i))));
                if(i < arr.length() - 1)
                    sb.append(sep);
            }
            sb.append("]");
            return sb.toString();
        }

        else {
            offset++;
            sb.append("[");
            for(int i = 0; i < arr.slices(); i++) {
                sb.append(format(arr.slice(i),rank - 1,offset));
                if (i != arr.slices() - 1) {
                    sb.append(",\n");
                    sb.append(StringUtils.repeat("\n",rank-2));
                    sb.append(StringUtils.repeat(" ",offset));
                }
            }
            sb.append("]");
            return sb.toString();
        }
    }
    {% endhighlight %}

The shape & buffer concept is extremely flexible from a computer science and mathematics perspective but difficult and obtuse from an end user data model perspective. The result is a well formed version of the N dimensional representation in the form of a string, but no data model in the base implementation that represents it without writing custom transformations that paginate the buffer. One solution is to use Jackson.

    {% highlight java %}
    /**
     * Weekly typed N dimensional List.
     */
    public static List makeRowsFromNDArray(INDArray source, int precision) throws IOException {
        String serializedData = new NDArrayStrings(", ",precision).format(source);
        List rows = (List)OBJECT_MAPPER.readValue(
                serializedData.getBytes(),List.class);
        return rows;
    }
    {% endhighlight %}

The problem with my approach is that its weakly typed, the user needs to know the depth of the dimensions to use it properly. Its also a little fat and slow.     

No utilities have been created by the skymind team to paginate the buffer recursively and create a simpler data model, and the closest thing is a jackson serializer that it is NOT recommend for pagination, even though writing a jackson de-serializer takes about on tenth the time. All you have to work with is the buffer and the shape.

![Not Jackson](/assets/images/not_jackson.png)

The founder is adamant that you **write your own** based on the original toString method and figuring out the transformation into a data model that you could do correlation on **is your problem noob, stop wasting my time**.

![Wasting my time](/assets/images/abusive_founder.png)

Wow. I never asked Adam for anything, never dumped **anything** on him. My crime was asking a forum of people if they had comments on an approach I took to get data out of an INDArray and if there was an alternative to using jackson that didn't me to write a whole bunch of code. He could have ignored me completely and this would have been a non issue. I am just a noob on a public forum with 50 users on it trying to learn how to use a technology and the next thing I know this fellow is DMing me telling me I am not respecting *his time*. It really took me back.

![Clay Response](/assets/images/response.png)

[Here is the transcript](/assets/txt/adam_dm.txt) of the entire DM exchange if you care.

## Implications of the BaseNDArray Implementation

The most important implications relate to the platforms ability to be used by "common folk" in the Enterprise. A simple hierarchical model that supported transformations of the paging recursively for N dimensional arrays would be a significant value add to the toolset.

A potential solution could be a nodal N level hierarchical model, something like nodes, and regression transformations that make it easy to get base models for correlation with labeling.

# Classification and Correlation

One of the most valuable use cases for the Enterprise for the use of deep learning is Classification and correlation. This would be the activity of taking a data set that you know something about.

What I found in the deeplearning4j/[dl4j-examples](https://github.com/deeplearning4j/dl4j-examples) was they were very good from a point of view of providing some sort of implementation for many common network approaches, and not so good in showing complete examples of how these approaches are likely to be used in the real world. Here is where this perspective comes from. Let's take the most basic example that a new user is like to try, CSVExample. This example shows the user how to read and train data, but fails at showing the potential user of how to correlate that back to the Classifications.

To solve for this I created a new example that correlated the Classifications back to the original test data, based on the classifications that the training data used.

```
0,Human
1,Cat
2,Dog
```

And then used the output of the network to figure out what the resulting classification was on the test data.

```
ANIMAL 0 was a Cat that lived 19 years and weighed 10 lbs. ate Mice and made the sound Meow.
ANIMAL 1 was a Dog that lived 9 years and weighed 60 lbs. ate Cats and made the sound Bark.
ANIMAL 2 was a Cat that lived 17 years and weighed 12 lbs. ate Mice and made the sound Meow.
ANIMAL 3 was a Dog that lived 9 years and weighed 50 lbs. ate Cats and made the sound Bark.
ANIMAL 4 was a Cat that lived 15 years and weighed 16 lbs. ate Mice and made the sound Meow.
ANIMAL 5 was a Dog that lived 9 years and weighed 70 lbs. ate Cats and made the sound Bark.
ANIMAL 6 was a Dog that lived 7 years and weighed 70 lbs. ate Cats and made the sound Bark.
ANIMAL 7 was a Dog that lived 10 years and weighed 40 lbs. ate Cats and made the sound Bark.
ANIMAL 8 was a Dog that lived 11 years and weighed 100 lbs. ate IceCream and made the sound Bark.
```

This implementation can be found here: [Animals Classifier example](https://github.com/deeplearning4j/dl4j-examples/blob/de860ca62206d63b796411ee3042efec10a861d0/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/BasicCSVClassifier.java)

## Common Flow in a dl4j-example

Let's look at the kind of example in dl4j, and break down what its accomplishes. Take a look at the [CSVExample]() and read along.

![flow one](/assets/images/correlation_1.png)

**Reading The Data**
In this example the all data for both training and testing is in one data set.

    {% highlight java %}
    DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
    DataSet allData = iterator.next();
    allData.shuffle();
    SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  
    {% endhighlight %}        

The implication of this choice is that it is difficult to correlate the output of the network to the testing dataset because there isn't a clear index to map the test data lines back the classification result from the output of the network. The test data is just 2/3 into the data set so there isn't a clear id for the test records you could use to map the results back to.

**Evaluation**

```
o.d.e.d.BasicCSVClassifier -
Examples labeled as 0 classified by model as 0: 8 times
Examples labeled as 0 classified by model as 1: 13 times
Examples labeled as 0 classified by model as 2: 23 times
```

What I would really :heart: is for the evaluation to tell me all the test row numbers and their classifications, something like [[1, "Normal"], [2,"Decreasing trend"]] because what it seems I need to do is to run some sort of test on the net output to determine the highest score for each row and then provide an implementation for the label mapping myself.


**Testing The Examples**

I have found one the best ways to teach developers what should be expected in terms of behavior is to create unit tests the explicitly test the outcomes you are expecting from your systems.

```
$ tree dl4j-examples/dl4j-examples/src -L 2
dl4j-examples/dl4j-examples/src
└── main
    ├── java
    └── resources
```

# Recommendations to Skymind

* Improve the Existing Libraries
    * Improve and simplify pagination, data ingress and egress utilities for ND4J
    * Enhance Label Mapping and Correlation Capabilities for Network Output and Evaluation
    * Provide unit tests that verify expected results based on datasets.
    * Correlation in all examples.
* Invest In Business Oriented Technical Documentation
* Hire a Technical Product Owner
* Create A Safe Space for Your Community
* Prioritize Customer Success Over Technical Elegance  

# Advice to the Founders
I am very sympathetic to you, more than you probably think. I was once young and believed that most people were not doing their part when it came to asking for help, and they aren't. But that is normal. Remember that if you *show no mercy* and are telling users that they are *wasting your time* and you threaten to ban them just because they asked a question that you are very likely alienating a potential customer. My honest recommendation is that you try to be more of a servant and instead of criticizing them, help them with compassion and kindness. If you can't do it because your personality doesn't permit it, hire someone who can and don't be customer facing. Wether you want to admit it or not you are killing deals before they ever begin.
