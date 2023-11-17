## The recipe for a successful tech-review channel

## Abstract

In the past few years, more and more people joined YouTube as content creators. Some of them managed to create a successful channel and others less so. We are interested in understanding why this happened (i.e. the reasons that “potentially” explain and led to the success of some channels over others). Based on an insightful data analysis conducted on the Youniverse dataset, we aim to formulate a guide, targeting new YouTubers/old YouTubers who would like to improve their channel in order to become more successful. Due to the large size of the dataset, we decided to focus on tech review channels.

## Research questions
The research questions that we would like to answer fall into 3 main topics as shown below:

1. General characteristics of successful tech review channels<br>
     For this topic, we would like to answer the following questions:
   
       1. Relationship between the length of a video and its success (how did it change over the years)?
   
       2. What should the upload frequency be?
   
       3. What is the right range of products to review (broad and narrow/niche product ranges)?
   
       4. Which tech review categories are the most popular (smartphone, laptop, …)?
      
2. Sentiment analysis<br>
     For this topic, we would like to answer the following questions:
   
       1. Compare positive/negative titles to see which one is attracting more views.
   
       2. Further sentiment analysis might be done on: 
        - length of title
        - question-like/non-question like titles
        - lower/capital case letters
   
3. Influence of big tech events (release) on channels' growth?<br>
     For this topic, we would like to answer the following questions:
   
       1. How do big tech events affect the existing channels growth?
   
       2. How quickly can the new channels (that start with the first video about the “big” event) grow?
   
       3. What is the best time delay for releasing a review video after a tech product launch event?
  
   
## Methods
### Step 1: General preprocessing
For our analysis, preprocessing includes filtering the Tech Review channels. We are doing it based on what percentageof the channels' videos are about tech review. To classify the tech review videos, howevere, we are using multiple techniques. The naive method is about finding what percentage of the words used in their title is widely used by the well-known Tech Review YouTubers, and classifying the video depending on this. We also want to try with the Bayesian classifier (as explained in the notebook) and other NLP methods in the next milestone.

### Step 2: Video length analysis
We focused on a subset of videos (namely videos shorter than 20 minutes long) from the tech review channels (identified in the previous step). We first computed the average number of likes and dislikes per video, as well as the average duration of a video, over the years. Then, we computed the ratios $\frac{number\ of\ likes}{number\ of\ views}$ and $\frac{number\ of\ dislikes}{number\ of\ views}$ and calculated their pearson correlations with the average duration. We found a statistically significant (small p-value) positive correlation between the ratio $\frac{number\ of\ likes}{number\ of\ views}$ and the average duration but a statistically significant negative correaltion between the ratio $\frac{number\ of\ dislikes}{number\ of\ views}$ and the average duration. Moreover, we computed the moving average with a window size of 1000 and finally, we computed the ratio $\frac{number\ of\ likes}{number\ of\ dislikes}$ and plotted it across years. Note that, we smoothed the number of views and number of dislikes by adding a one to each of them, so that we avoid dividing by zero in our ratio computations while still taking into account those videos that performed poorly instead of discarding them.

### Step 3: Upload frequency analysis
First we tried naively to compute the frequency as $\frac{number\ of\ videos}{number\ of\ months}$. 
Then second thing we’ve done is to check what’s the average frequency for each category (in terms of number subscribers).
Then we desgined a particular way of frequency : first putting number of days between video releases, and then putting $\frac{1}{number\ of\ days}$ (when number of days between videos is 0. We assumed that 2 videos are uploaded the same day and replaced 0 by 1/2).
We then calculated the variance of this frequency (to check whether the upload is consistent or not). Aftewards, we compared the number of videos per month with the delta subs this month (computing correlation) and then by comparing $\frac{number\ of\ videos\ per \ month}{var(freq)}$ with delta subs we found a higher correlation, specially for smaller youtubers.

### Step 4: Influence of big tech events on channel growth
We identified the channels that talk about a release event to analyze their evolution in terms of the number of subscribers compared to those that do not talk about the event. First, we focus on 1 event (the release of the iPhone X). Then, we will generalize to 5 others. From the title and tags of the videos, we classify them according to the presence of the item "iphone x" in this metadata : treat if it talks about the release - control if not. We focus on videos that have an upload date from the release till one month later.
Then, we classify the channels considering that if a channel has at least one video that talks about the release, then it is a treat channel. 
We compare the growth of channels (measured in terms of the number of subscribers) for treat and control channels to see if channels that talk about a release have a higher growth than the ones that do not. To do this comparison, we use the time series data and focus on a period that goes from 15 days before the release till 15 days after.
Then, we will do an observational study (causal analysis) with the outcome being the number of subscribers while identifying the potential confounders.
### Step 5: Sentiment analysis
In the section of the project, our aim is to study the titles of videos from tech YouTube channels to come up with well thought rules about not only subjects that could be discussed by a tech channel, but also how to write a title that would attract the most viewers.
For instance, we would answer questions such as how should be the overall sentiment of the title, what should be the overall length …


**Note: more implementation details/explanations can be found in the notebook.**

## Proposed timeline
```
.
|── 17.11.2023 - Milestone 2 deadline
|
├── 21.11.2023 - Continue exploring the dataset 
│  
├── 01.12.2023 - Homework 2 deadline
│    
├── 05.12.2023 - Non-naive preprocessing (classification)
│  
├── 12.12.2023 - Causal analysis on events + Develop draft for data story
│  
├── 18.12.2023 - Sentiment analysis
│  
├── 21.12.2023 - Finalize data story page design
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
    <td class="tg-0lax">(Step 4) Work more on question 3.1, by generalizing to other events, and do the causal analysis.<br><br>(Step 4) Do 3.2.<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Jakhongir</td>
    <td class="tg-0lax">...<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Zied</td>
    <td class="tg-0lax">(Step 5) Work on the sentiment analysis topic (Do 2.1)<br><br>(Step2) Continue exploring the dataset.<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Ali</td>
    <td class="tg-0lax">...<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Othmane</td>
    <td class="tg-0lax">(Step 3) Work on the upload frequency: objective is to generalise the study to all youtubers instead of 4<br><br>(Step 3) Run a causal study on the upload frequency: split youtubers into different chunks as it has been done in the beginning of the frequency analysis<br><br>Help in writing the report/data story + website (presentation of data story)</td>
  </tr>
</tbody>
</table>
