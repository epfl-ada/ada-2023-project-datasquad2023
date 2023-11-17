## The recipe for a successful tech-review channel

## Abstract

In the past few years, more and more people joined YouTube as content creators. Some of them managed to create a successful channel and others less so. We are interested in understanding why this happened (i.e. the reasons that “potentially” explain and led to the success of some channels over others). Based on an insightful data analysis conducted on the Youniverse dataset, we aim to formulate a guide, targeting new YouTubers/old YouTubers who would like to improve their channel in order to become more successful. Due to the large size of the dataset, we decided to focus on tech review channels.

## Research questions
The research questions that we would like to answer fall into 3 main topics as shown below:

1. General characteristics of successful tech review channels
     For this topic, we would like to answer the following questions:
   
       1. Relationship between the length of a video and its success (how did it change over the years)?
   
       2. What should the upload frequency be?
   
       3. What is the right range of products to review (broad and narrow/niche product ranges)?
   
       4. Which tech review categories are the most popular (smartphone, laptop, …)?
      
2. Sentiment analysis
     For this topic, we would like to answer the following questions:
   
       1. Compare positive/negative titles to see which one is attracting more views.
   
       2. Further sentiment analysis might be done on: 
        - length of title
        - question-like/non-question like titles
        - lower/capital case letters
   
3. Influence of big tech events (release) on channels' growth?
     For this topic, we would like to answer the following questions:
   
       1. How do big tech events affect the existing channels growth?
   
       2. How quickly can the new channels (that start with the first video about the “big” event) grow?
   
       3. What is the best time delay for releasing a review video after a tech product launch event?
  
   
## Methods
### Step 1:
### Step 2:
### Step 3:
### Step 4:
### Step 5:
### Step 6:
Identify the channels that talk about a release event to analyze their evolution in terms of the number of subscribers compared to those that do not talk about the event. First, we focus on 1 event (the release of the iPhone X). Then, we will generalize to 5 others. From the title and tags of the videos, we classify them according to the presence of the item "iphone X" in this metadata : treat if it talks about the release - control if not. We focus on videos that have an upload date from the release till one month later.
Then, we classify the channels considering that if a channel has at least one video that talks about the release, then it is a treat channel. 
We compare the growth of channels (measured in terms of the number of subscribers) for treat and control channels to see if channels that talk about a release have a higher growth than the ones that do not. To do this comparison, we use the time series data and focus on a period that goes from 15 days before the release till 15 days after.
Then, we will do an observational study (causal analysis) with the outcome being the number of subscribers while identifying the potential confounders.

## Proposed timeline
```
.
|── 17.11.2023 - Milestone 2 deadline
|
├── 21.11.2023 - Perform sentiment analysis
│  
├── 23.11.2023 - Continue exploring the dataset to get more insights
│  
├── 01.12.2023 - Homework 2 deadline
│    
├── 05.12.2023 - Perform final analysis
│  
├── 12.12.2023 - Develop draft for data story
│  
├── 18.12.2023 - Finalize code implementations and visualizations
│  
├── 21.12.2023 - Finalize data story
│  
├── 22.12.2023 - Milestone 3 deadline
.
```

## Organization within the team

<table class="tg" style="undefined;table-layout: fixed; width: 342px">
<colgroup>
<col style="width: 164px">
<col style="width: 178px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">Tasks</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">@Salma</td>
    <td class="tg-0lax">Work on question 3.1, generalize to the other events, and do the causal analysis.<br><br>Do 3.2.<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Jakhongir</td>
    <td class="tg-0lax">...<br><br>Develop the web interface</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Zied</td>
    <td class="tg-0lax">Work on the sentiment analysis topic. Do 2.1<br><br>Continue exploring the dataset.<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Ali</td>
    <td class="tg-0lax">...<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Othmane</td>
    <td class="tg-0lax">...<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
</tbody>
</table>
