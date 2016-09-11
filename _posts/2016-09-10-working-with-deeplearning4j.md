---
layout: post
title:  "A newbie uses deeplearning4j"
date:   2016-09-10 14:42:30 -0700
categories: deeplearning java
---

I admit it. I am ignorant. I also am not a wizard, information doesn't just get injected into my brain as if I was neo from the matrix. **I know kungfu**.
So in my mind its normal to ask dumb questions or misunderstand when coming to something new and fresh. Helping people be successful means being compassionate when people come to you and ask for help. That's where I am coming from with this post in my first impressions and issues I am seeing around using **deeplearning4j**.

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

![Wasting my time](/assets/images/abusive_founder.png)

## Implications of the BaseNDArray Implementation




# Recommendations



```
Adam Gibson @agibsonccc Sep 09 17:02
FWIW
I crack down on people who don't do research
Scroll up
You'll see where the guy didn't even know what a gpu was or how it worked
You don't need to be humble trust me
Just ask away
I wish we had more people like you frankly
"Help me understand something, I've done some research" isn't "Do it for me lol I don't respect your time"
Ease up a bit :wink:

Clay Graham @claytantor 09:26
Appreciate the sentiment. I attempt to have humility because for much of my life it was missing. It is important to show those, who I am asking help from, to know that I appreciate anything they provide. Hope you understand.
But I also understand that I have to make an investment in understanding. RTFD is an acceptable answer, if the docs actually have the answer of course.

Adam Gibson @agibsonccc 09:26
Well so what are you missing here?
I asked you to look at the source code
not use jackson
I already said we didn't have the arrays stuff you were looking for

Clay Graham @claytantor 09:27
Well I guess I am lazy. I want an easier way if possible. :smiling_imp:

Adam Gibson @agibsonccc 09:27
but we implemented it in our to string stuff
Don't dump that on me
If you deviate I have zero mercy for you
That's frankly wasting my time
I gave you that advice because I know what's there

Clay Graham @claytantor 09:28
Of course I am not dumping anything on you sir, sorry if you took offence.

Adam Gibson @agibsonccc 09:28
Well no it just doesn't matter
It wastes both of our time
I'm trying to show you what you want

Clay Graham @claytantor 09:28
I am sorry you feel that way.

Adam Gibson @agibsonccc 09:28
despite not understanding why you even need to do this
So again here's the looping logic
Go here and look through this

Clay Graham @claytantor 09:28
ok. so sorry. I will leave this group then. sorry to offend.

Adam Gibson @agibsonccc 09:29
https://github.com/deeplearning4j/nd4j/blob/master/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/string/NDArrayStrings.java
You don't need to leave
Just listen to us a bit
I'm just making things abundantly clear where things need to go
We get along fine if you can just cooperate a bit
"Interpret what you said and attempt to do something off" isn't helping anyone :wink:
Look up to you what you decide to do
I know you're learning we all are in some ways

Clay Graham @claytantor 09:46
I am happy to listen, but I have to be honest that telling a potential user that they are wasting your time is not going to work for you as a founder. Its abusive, and will push people away. I am ignorant, so I am going to ask stupid questions. You are going to get way stupider questions than mine and if your reaction is the one you just gave me, I am telling you, its not going to work out well. When you get frustrated with people who are aren't as smart as you, and you will over and over because you are very smart, you will need to see yourself as a servant if you want to be successful.

Adam Gibson @agibsonccc 09:49
Look again - up to you. I've sat for a long time with some people
I grew this to as large as what it is because of how long I sat here
Part of that was convincing people to try something and iterate on it
If you look the conversation afterwards was fairly productive - we talk about trade offs of things
Ive seen nice people and people who treat us like doormats
I have a firm middle ground I take on this nothing more
Up to you if you want to try again - all I ask is you give the approaches we give you a try

Adam Gibson @agibsonccc 09:54
It'll focus things on being productive. There's a lot of rabbit holes to go down

```
