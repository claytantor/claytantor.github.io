---
layout: post
title:  "Response to deeplearning4j's AlexDBlack."
date:   2016-09-19 19:42:30 -0700
categories: deeplearning java
---


AlexDBlack,

Thanks so much. I am honored that you are curious about my perspective.

Let me start by saying that I am very sympathetic to the challenges of a startup. I have been a founder more than once so I am sure its tough to do everything you want to do. Building software for the enterprise is especially difficult, the body of work needed to make meet the expectations of larger companies can be daunting.

The technology you have has significant value and has the potential of making deep learning a tool for data scientist and analysts in corporations. A pure deep learning implementation with an eye for performance in Java is something we need, and I would recommend using it even at this early stage. I made sure to put that in my review.  

So, good work. Keep going.

My review is meant to explain what kind of speed bumps that users in the enterprise are likely to experience. No to roast your company.

>our docs are a mixture of pretty good, 'kinda sucks' and non-existent at the moment, depending on the area

I have found that "Quick-Start" Guides that get a user running, but understanding a technology fast is helpful. My recommendation would be to go beyond that and create a user's guide for each technology where the core aspects of building data, training, using the network and correlation are broken down and explained, with working code.

Think about who your users are:

* They use databases, traditional ORM models are going to be the most common data source.
* The engineers most likey using this have never studied deep learning, they may have experience with map reduce, but most likely are building Hive pipelines.
* They are not rock stars, they are quite possibly not even senior.
* They have never written foundational code.

This means you need to focus on making it easy to get data in and out and to show the full path of starting with data in grid like structures and getting a meaningful result in a grid like format.

>To be fair, ND4J is necessarily complex due to the large amount of functionality there..

Don't remove ANY of that, its awesome, but build some bridges that marshal in and out so people who have a hard time understanding it can use it quickly.

**People just love grids**, even if they aren't awesome for deep learning. A simple guava to nd4j bridge, or even some examples that used guava and showed how to get data in and out could be very helpful to a common enterprise developer. I am pretty sure that many many people will want to so something like this:

```
public DataSet getSeriesValuesForSegmentsList(Table<Date,String,Double> measureForDateRange)
            throws DataSetProviderException
    {
        Map<String,List<Double>> dataListMap = new HashMap<>();

        //make the buffer for each
        measureForDateRange.columnKeySet().forEach(ticker->{
            dataListMap.put(ticker,new ArrayList<>());
        });

        // we dont know what dates exist for each value so we need a
        // way to normalize values in the series
        Set<Date> commonDates = new HashSet<>();
        List<Date> dates = new ArrayList<Date>(measureForDateRange.rowKeySet());
        Collections.sort(dates);
        dates.forEach(date->{
            Map<String,Double> cols = measureForDateRange.row(date);
            if(cols.keySet().size()==segments.size()){ //all series values exist
                commonDates.add(date);
                segments.forEach(ticker->{
                    List<Double> dataForTicker = dataListMap.get(ticker);
                    dataForTicker.add(measureForDateRange.get(date,ticker));
                });
            }
        });

        //make one list for data
        List<Double> alldata = new ArrayList<Double>();
        dataListMap.keySet().forEach(ticker->{
            alldata.addAll(dataListMap.get(ticker));
        });

        INDArray segmentsArray = Nd4j.create(
                this.convert(alldata),
                new int[]{
                        segments.size(),
                        commonDates.size()});

        // make fake labels, this should be a training schema
        float[] labels = new float[segments.size()];
        for (int i = 0; i <segments.size() ; i++) {
            labels[i] = 1;
        }
        INDArray labelsArray = Nd4j.create(labels, new int[]{segments.size(),1});

        DataSet ds = new DataSet(segmentsArray, labelsArray);

        return ds;

    }
```    

Sorry that example isn't complete, I hope it gives the gist though.

>mapping evaluation/results back to the raw data - yep. We've talked about this internally, we've had higher priorities

That's totally ok. They are your priorities, and your business. Its not my job to set your priorities because I bet you are juggling chainsaws. I am just an enthusiast trying to introduce deep learning to Nike. I would say this though, your examples (at least right now) don't prove that they work at runtime to someone looking at them. There is no clear result of the network, that reduces significantly their value as a teaching and evaluation tool.

Hope this helps,

Clay

>
