# A recipe for a successful tech-review channel

## Datastory 
Uncover the secret to creating the perfect recipe for a successful tech-review channel on YouTube! Follow the link to our datastory and discover the ingredients for success: [Datastory](https://jakhongir0103.github.io/datastory/) 

## Abstract

In the past few years, more and more people joined YouTube as content creators. Some of them managed to create a successful channel and others less so. We are interested in understanding why this happened (i.e. the reasons that “potentially” explain and led to the success of some channels over others). Based on an insightful data analysis conducted on the Youniverse dataset, we aim to formulate a guide, targeting new YouTubers/old YouTubers who would like to improve their channel in order to become more successful. Due to the large size of the dataset, we decided to focus on tech review channels.

## Research questions
The research questions that we would like to answer fall into 3 main topics as follows:
(For each topic, we state the questions we would like to answer below it)

1. General characteristics of successful tech review channels<br>

       1. What is the optimum duration of a tech review video?
   
       2. What should the upload frequency of a youtuber be?
   
       3. How does the review product type influence the channels' growth?
      
2. Attracting viewers by title<br>
   
       1. How should the overall sentiment of a video's title be?
   
3. Big tech product releases<br>
   
       1. Does releasing a video about a newly released tech product influence the channels' growth?
   
       2. What topics, of videos before and after the product release, accelarate the channels' growth?
  
   
## Methods
### General preprocessing
For our analysis, preprocessing includes filtering the Tech Review channels. We are doing it based on what percentage of the channels' videos are about tech review. To classify the tech review videos, the method is about finding what percentage of the words used in their title is widely used by the well-known Tech Review YouTubers, and classifying the video depending on this.

### 1.1
We divided the tech videos into 2 subsets according to its duration: below 20 minute, and above 20 minute. For each groups, we then computed the correlation of the length of a video to multiple metrics, namely $\frac{number\ of\ likes}{number\ of\ views}$, $\frac{number\ of\ dislikes}{number\ of\ views}$, $\frac{number\ of\ likes}{number\ of\ dislikes}$, and $\frac{number\ of\ likes}{number\ of\ dislikes}$.

### 1.2
Here we definde a monthly regularity of a channel as:

$$
\text{number of videos} \cdot \log\left(1 + \frac{1}{\text{frequency}_{\text{std}}}\right)
$$

where ${\text{frequency} = \frac{1}{\text{delay}}}$, and _delay_ is number of days between publishing 2 consecutive videos.

Then we find the correlation between the regularity and the growth of the channel, where the growth is defined as $\text{Channel's growth}$ as ${\frac{\text{number of monthly new subscribers}}{\text{number of total subscribers}}}$.

### 1.3
We first classify videos into one of the 7 types we are analysing (laptop, phone, camera, headphone, smart watch, tablet, desktop setup), depending on some predefined [keywords](data\product_keywords.json) that are relevant to these tech types. Then for each channel, we are calculating the percentage of each product type videos and aswer the following questions:
- What range of product types should be covered?
- What product categories have higher influence on the channels growth?
- Which product categories attract more viewers?

### 2
In the section of the project, we calculated the overall sentiment of the titles using 2 methods ([textblob](https://textblob.readthedocs.io/en/dev/), [vaderSentiment](https://www.nltk.org/api/nltk.sentiment.vader.html)). Then we duscussed its relation to the videos success within the channels, i.e. # views of a video / # subs of a channel.

### 3.1
We analysed the reaction of viewers during multiple periods of the release date of multiple products. We first seperate the channels into 2 categories:
- The ones that have published a video about a product we are analyzing at hand.
- The ones that have NOT published a video about a product we are analyzing at hand.

Then, we compare the trend of the channel's growth over a 1 year period (6 months before and after the release of the product) to see if there is a clear distinction between them.

### 3.2
Here, we saw the topics that are the most discussed during 3 periods around the release date of iPhone X: pre, during and post. Then we dove deeper into the analysis of those channels, that discussed the product during the pre release period, to see how it helped their growth.


**Note: more implementation details/explanations can be found in the notebook.**

## Executed timeline
```
.
├── 01.12.2023 - Homework 2 deadline
│    
├── 05.12.2023 - Reformulating the research questions and digging deeper into them 
│  
├── 12.12.2023 - Causal analysis on events
│  
├── 18.12.2023 - Sentiment analysis + Develop draft for data story
│  
├── 21.12.2023 - Finalize data story page design as well as the notebook
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
    <td class="tg-0lax">Work more on question 3.1, by generalizing to other events, and do the causal analysis.<br><br>Data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Jakhongir</td>
    <td class="tg-0lax">Implement preprocessing<br><br>Work on question 1.3.<br><br>Data story + website</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Zied</td>
    <td class="tg-0lax">Work on the sentiment analysis topic (2.1)<br><br>Work on question 1.1.<br><br>Data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Ali</td>
    <td class="tg-0lax">Work on the sentiment analysis topic (2.1)<br><br>Work on question 3.1.<br><br>Data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@Othmane</td>
    <td class="tg-0lax">Work on the upload frequency (1.2)<br><br>Data story</td>
  </tr>
</tbody>
</table>
