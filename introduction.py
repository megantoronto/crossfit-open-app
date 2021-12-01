import streamlit as st

def app():
    st.title("CrossFit Open Dashboard")
    st.header("Introduction")

    st.markdown('''This dashboard was built to analyze CrossFit Open data from 2011-2021. The data consists of two main sources:

- **CrossFit Open Leaderboard Data**
    - CrossFit Open Workout Scores
    - Demographic data
- **CrossFit Open Workouts**
    - Movements
    - Rep schemes
    - Time domains
    - Loading
    
Each tab analyzes a different aspect of the CrossFit Open.

## Demographics
The Demographics tab analyzes the following information:
- CrossFit Open participation numbers
- Age Category breakdown
- Height breakdown
- Weight breakdown
- Can filter each breakdown by overall CrossFit Open rank to see what the composition looks like for the top 50, 100, etc

**Data caveats:**
- I was only able to pull the leaderboard data for people from age 16-54. The 14-15 and 55+ age categories are hosted on a separate leaderboard that I did not have a chance to pull. The total participation numbers are therefore caveated for people aged 16-54.
- Pre 2017, the age categorization for the 16-17 years olds was not very good. The 16-17 age category pre 2017 is not the most reliable data and is likely not a true representation of the number in that age category that participated. I hope at a later date to get the rest of the age category data for each year to have the most accurate counts for each age category.
- Height and weight are self reported values, so take these numbers with a grain of salt. I also had to add in some filtering for values that were clearly incorrect and some other editing, so use this analysis as a general representation rather than a source of truth.

## Movements
The Movements tab analyzes the following information
- Shows all movements that have appeared in the CrossFit Open
- Allows you to filter by a single movement to see:
    - Which workouts that movement appeared in
    - What movements it is most commonly paired with
    - The historical weights for weighted movements
    - The amount of reps the Top 50, Top 500, and Top 10,000 typically perform year over year of that movement
    
## Workouts
The Workouts tab analyzes the following information:
- The count and categorization of the general types of workouts that have appeared in the CrossFit Open
- An overview of the time domains for CrossFit Open workouts

**Data caveats:**
- I view the time domains as the maximum amount of time a workout can take. We know a 20 minute AMRAP lasts 20 minutes, but a For Time workout with a 15 minute time cap does not necessarily last 15 minutes. In the future I would like to add an analysis here on the average time For Time workouts take to have a better estimate of time domain.
- There are a few workouts that have no time caps, or are to failure workouts with no maximum time set. These are not included in the time domain analysis. In a future analysis, I plan to pull in what the average time spent is on the workouts to get an estimate of what the time domain for these workouts is.

## Workout Performance
The Workout Performance tab analyzes the following information:
- Allows you to filter by:
    - Workout
    - Gender
    - Number of Athletes
    - Workout Rank vs Overall Rank (Do you want to see the performance of the top 50 in the workout or the performance of the overall top 50 in the workout?)
- After filtering, a scores table will show the name, score, and details for however many athletes you pulled in
- Additionally, another table will show the average score/information for however many athletes you pulled in
- Because many workouts are so different, different information will appear depending on the workout type and structure

**Data caveats:**
- Pre 2017 leaderboard data can have some holes in it. The farther you try to query down those leaderboards (>10,000) the more data issues that can arise. The Top 10,000 rows for each year appear to be very solid.
- 15.1a is missing some scores which for some reason cannot be found on games.crossfit.com.

## General Notes
- This app only analyzes performance and movement data based on people who have performed the workout as prescribed. There is no current analysis or inclusion of scaled workout data at the moment.
- The workouts that have shown up in the Open as combos (15.1/15.1a, 18.2/18.2a, 21.3/21.4) are currently all being treated as separate workouts. At a later date I would like to find a better way to represent how these workouts interact with each other. 
    ''')