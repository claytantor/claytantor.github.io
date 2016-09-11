---
layout: post
title:  "A newbie uses deeplearning4j"
date:   2016-09-10 14:42:30 -0700
categories: deeplearning java
---

I admit it. I am ignorant. I also am not a wizard, information doesn't just get injected into my brain as if I was neo from the matrix. **I know kungfu**.
So its normal to ask dumb questions or misunderstand when coming to something new and fresh. That's where I am coming from with this post in my first impressions and issues I am seeing around using **deeplearning4j**.

Most enterprise companies understand this and spend a ton of time, money and energy on building documentation, cookbooks and how-tos that make understanding the essentials of how a technology is used for end to end use cases possible.

The commercial company behind **deeplearning4j** is [skymind](https://skymind.io/). It positions itself as "Deep Learning for Enterprise Level Applications", which is and important claim because enterprise is not focused on science, or mathematics, its focused on business impact. Business concerns are different than academic ones and as a set of technologies like this transition from primarily academic settings to commercial ones the product owner should spend time and have concern for.

This post is about my experiences in the first week of using deeplearning4j, the challenges I faced and my experiences.


## Important Links

* [All deeplearning4j Repos](https://github.com/deeplearning4j) - This is all the repos related to deeplearning4j, including the examples.
* [ND4j Repo On Github](https://github.com/deeplearning4j/nd4j) - ND4J is the mathematics foundation classes used by deeplearning4j.  
* [Andy Gibson's Glitter](https://gitter.im/agibsonccc) - *Newbie's beware.* Ask the wrong question to the group and Andy will start a DM with you and tell you that you are wasting his time. I found that AlexDBlack was helpful even though I asked him one very dumb question.

## Classification
I would argue that a giant business use cases is about automation of classification. Making simple data structures that make it easy to build recordsets, and then to simply get the classifications for those records automatically is important to broad adoption.

When trying to use **deeplearning4j** I ran into some specific challenges that slowed me down.

### Using INDArray
INDArray is a buffer, and its really great at being flexible for many different use cases because it can be used in N dimensional problems. All you need to do is use the shape to then split the pages up.

https://www.mathworks.com/help/matlab/math/multidimensional-arrays.html?requestedDomain=www.mathworks.com&requestedDomain=www.mathworks.com

![Buffer Pagination](https://www.mathworks.com/help/matlab/math/array_storage.gif)

The data buffer in this example is a single array.

`[0.3,2.4,3.6,1.3,3.4,1.6,2.3,0.4,0.6,3.3,1.4,0.6]`

and the shape: `[3,4]`

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

The result is a well formed version of the N dimensional representation in the form of a string, but no data model in the base implementation that represents it without writing transformations that paginate the buffer.

No utilities have been created to paginate the buffer, and the closest thing is a jackson serializer that they do NOT recommend that you use for pagination.



All you have to work with is the buffer and the shape. The recommendation from the team is to **write your own** and figuring out the transformation into a data model that you could do correlation on **is your problem noob, stop wasting my time**.

![Image of Yaktocat](/images/abusive_founder.png)

## Implications of the BaseNDArray Implementation




# Recommendations
