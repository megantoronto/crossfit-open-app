import streamlit as st
import psycopg2
#import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
from datetime import timedelta


def calc_table_height(df, base=208, height_per_row=30, char_limit=30, height_padding=16.5):
    '''
    df: The dataframe with only the columns you want to plot
    base: The base height of the table (header without any rows)
    height_per_row: The height that one row requires
    char_limit: If the length of a value crosses this limit, the row's height needs to be expanded to fit the value
    height_padding: Extra height in a row when a length of value exceeds char_limit
    '''
    total_height = 0 + base
    for x in range(df.shape[0]):
        total_height += height_per_row
    for y in range(df.shape[1]):
        if len(str(df.iloc[x][y])) > char_limit:
            total_height += height_padding
    return total_height

def flatten_list(x):
    return [i for c in x for i in c]

def gen_table_colors(df,rowEvenColor,rowOddColor):
    table_colors=[]
    for i in range(len(df)):
        if i % 2==0:
            table_colors.append(rowEvenColor)
        else:
            table_colors.append(rowOddColor)
    return table_colors

def create_conn():
        conn = psycopg2.connect(
            **st.secrets["postgres"])
        return conn

#@st.cache()
def load_data(table):
    conn = create_conn()
    query = '''SELECT * FROM ''' + table
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def load_result_data(gender,year,workout,rank):
    conn = create_conn()
    query = '''SELECT rank_'''+str(workout)+''', competitorname, scoredisplay_'''+ str(workout)+''', breakdown_''' + str(workout)+''' FROM open_'''+str(year)+'''_'''+gender+ \
    ''' WHERE scaled_''' +str(workout)+''' IN ('0','false') ORDER BY rank_''' +str(workout) +''' LIMIT '''+str(rank)
    data = pd.read_sql(query, conn)
    #conn.close()
    return data

def format_time(x):
    x_formatted = ':'.join(str(x).split(':')[1:])
    return x_formatted


def calc_total_reps(workout,score_data,df_rep,workout_num,gender,special):
    #df_rep= load_data('rep_rounds')
    df = df_rep[df_rep['workout']==workout]
    df_non_null = df.dropna(axis=1,how='all')
    m_cols = [c for c in df_non_null.columns if "movement" in c and "reps" not in c and "addition" not in c]
    cols =  [c for c in df_non_null.columns if "movement" in c and "reps" in c and "addition" not in c]
    movements=[df[c].values[0] for c in m_cols]
    reps = [df[c].values[0] for c in cols]
    if (workout =="18.1") & (gender=="Women"):
        reps=[8,10,12]
    d={}
    for i in range(0,len(movements)):
        if movements[i] not in d:
            d[movements[i]]=0
        d[movements[i]]+=reps[i]
    scores=score_data['scoredisplay_'+str(workout[workout.find(".")+1:])].values
    final_dict ={}
    for i in  movements:
        if i != "rest":
            final_dict[i]=[]
    if workout=="17.2":
        time_domain = timedelta(minutes=int(df['time_domain'].values[0]))
        vals = d.values()
        total_reps = sum(reps[0:3])
        final_dict['rounds']=[]
        final_dict['reps']=[] 
        final_dict['avg_time_per_round']=[]
        for i in range(0,len(movements)):
            if movements[i] not in final_dict:
                final_dict[movements[i]]=[]
        for score in scores:
            print(score)
            if "reps" in score:
                score = int(score[:score.find(" ")])
            else:
                score=int(score)
            full_rounds = int(score//total_reps)
            remainder = round((score/total_reps-full_rounds)*total_reps)
            time_per_round = str(time_domain/score*total_reps)
            time_per_round = ':'.join(time_per_round.split(':')[1:])
            final_dict['rounds'].append(full_rounds)
            final_dict['reps'].append(remainder)
            final_dict['avg_time_per_round'].append(time_per_round)
            tracker_dict={}
            for i in range(0,len(movements)):
                if movements[i] not in tracker_dict:
                    tracker_dict[movements[i]]=0
            count=0
            switch=True
            for i in range(full_rounds):
                if i in [0,1,4,5]:
                    switch=True
                else:
                    switch=False
                if switch:
                    tracker_dict['dumbbell_front_rack_walking_lunges']+=10
                    tracker_dict['toes_to_bar']+=16
                    tracker_dict['dumbbell_power_clean']+=8
                else:
                    tracker_dict['dumbbell_front_rack_walking_lunges']+=10
                    tracker_dict['bar_muscle_up']+=16
                    tracker_dict['dumbbell_power_clean']+=8
                
                count+=1
            if remainder>0:
                if switch:
                    for i in range(6,9):
                        if remainder - reps[i]>0:
                            tracker_dict[movements[i]]+=reps[i]
                        else:
                            tracker_dict[movements[i]]+=remainder
                            break
                        remainder=remainder-reps[i]
                else:
                    for i in range(0,len(movements[0:3])):
                        if remainder - reps[i]>0:
                            tracker_dict[movements[i]]+=reps[i]
                        else:
                            tracker_dict[movements[i]]+=remainder
                            break
                        remainder=remainder-reps[i]
            final_vals=list(tracker_dict.values())
            for i in range(0,len(final_vals)):
                final_dict[list(tracker_dict.keys())[i]].append(final_vals[i])
    elif workout == "20.5":
        time_domain = timedelta(minutes=int(df['time_cap'].values[0]))
        bd=score_data['breakdown_'+str(workout[workout.find(".")+1:])].values
        movements.append('tiebreak')
        final_dict['tiebreak']=[]
        for b in bd:
            if "240" not in b:
                m = b.split("\n")
                for i in range(0,len(m)):
                    if i in (0,2):
                        val = int(m[i][:m[i].find(" ")])
                    elif i == 1:
                        val = int(m[i][:m[i].find("-cal")])
                    else:
                        val = m[i][m[i].find(" ")+1:]
                    final_dict[movements[i]].append(val)
            else:
                final_dict['ring_muscle_up'].append(40)
                final_dict['row_calorie'].append(80)
                final_dict['wall_ball'].append(120)
                final_dict['tiebreak'].append("")
        total_reps = 240
    elif (df['type'].values[0]=="AMRAP") & pd.isnull(df['movement_1_rep_addition'].values[0]):
        time_domain = timedelta(minutes=int(df['time_domain'].values[0]))
        vals = d.values()
        total_reps = sum(vals)

        final_dict['rounds']=[]
        final_dict['reps']=[] 
        final_dict['avg_time_per_round']=[]           
        for score in scores:
            if "reps" in score:
                score = int(score[:score.find(" ")])
            else:
                score=int(score)
            full_rounds = int(score//total_reps)
            remainder = round((score/total_reps-full_rounds)*total_reps)
            time_per_round = str(time_domain/score*total_reps)
            time_per_round = ':'.join(time_per_round.split(':')[1:])
            final_dict['rounds'].append(full_rounds)
            final_dict['reps'].append(remainder)
            final_dict['avg_time_per_round'].append(time_per_round)
            x=[c*full_rounds for c in d.values()]
            f=[]
            for i in d.values():
                if remainder-i>0:
                    f.append(i)
                else:
                    f.append(remainder)
                    check = len(d.values())-len(f)
                    if check !=0:
                        for i in range(check):
                            f.append(0)
                    break
                remainder=remainder-i
            final_vals = np.array(x)+np.array(f)
            for i in range(0,len(final_vals)):
                final_dict[list(final_dict.keys())[i]].append(final_vals[i])
    elif (df['type'].values[0]=="AMRAP") or (df['type'].values[0]=="to_failure"):
        total_reps=0
        if pd.notnull(df['time_domain'].values[0]):
            time_domain = timedelta(minutes=int(df['time_domain'].values[0]))
        else:
            time_domain=0
        movement_1_rep_addition = df['movement_1_rep_addition'].values[0]
        movement_1_rep_addition_when =  df['movement_1_rep_addition_when'].values[0]
        movement_2_rep_addition = df['movement_2_rep_addition'].values[0]
        movement_2_rep_addition_when =  df['movement_2_rep_addition_when'].values[0]
        scores=score_data['scoredisplay_'+str(workout[workout.find(".")+1:])].values
        for score in scores:
            if "reps" in score:
                score = int(score[:score.find(" ")])
            else:
                score = int(score)
            m1_tracker=df['movement_1_reps'].values[0]
            m2_tracker=df['movement_2_reps'].values[0]
            movement_1_val=0
            movement_2_val=0
            movement_1_counter=0
            movement_2_counter=0
            while score>0:
                if movement_1_counter==movement_1_rep_addition_when:
                    m1_tracker+=movement_1_rep_addition
                    if score-m1_tracker>0:
                        movement_1_val+=(m1_tracker)
                        score -= m1_tracker
                        movement_1_counter=0
                    else:
                        movement_1_val+=score
                        break
                else:
                    if score-m1_tracker>0:
                        movement_1_val+=m1_tracker
                        score -= m1_tracker
                    else:
                        movement_1_val+=score
                        break
                movement_1_counter+=1   
                #print("1: ",movement_1_val)
                #print("M1 Tracker:",m1_tracker)
            
                if movement_2_counter==movement_2_rep_addition_when:
                    m2_tracker+=movement_2_rep_addition
                    if score-m2_tracker>0:
                        movement_2_val+=m2_tracker
                        score -= m2_tracker
                        movement_2_counter=0
                    else:
                        movement_2_val+=score
                        break
                        
                else:
                    if score-m2_tracker>0:
                        movement_2_val+=m2_tracker
                        score -= m2_tracker
                    else:
                        movement_2_val+=score
                        break
                movement_2_counter+=1

                #print("2: ",movement_2_val)
                #print("M2 Tracker:",m2_tracker)
            final_vals=[movement_1_val,movement_2_val]
            for i in range(0,len(final_vals)):
                final_dict[list(final_dict.keys())[i]].append(final_vals[i])
    elif (df['type'].values[0]=="for_time") & pd.notnull(df['rounds'].values[0]):
        vals = d.values()
        total_reps = sum(vals)
        rounds= df['rounds'].values[0]
        time_domain = timedelta(minutes=int(df['time_cap'].values[0]))
        scores=score_data['scoredisplay_'+str(workout[workout.find(".")+1:])].values
        #final_dict = {}
        final_dict['avg_time_per_round']=[]
        for score in scores:
            if "reps" not in score:
                score = timedelta(minutes=int(score[:score.find(":")]),seconds=int(score[score.find(":")+1:]))
                final_dict['avg_time_per_round'].append(format_time(score/rounds))
                final_vals = [c*rounds for c in d.values()]
                for i in range(0,len(final_vals)):
                    final_dict[list(final_dict.keys())[i]].append(final_vals[i])
            else:
                score = int(score[:score.find(" ")])
                full_rounds = int(score//total_reps)
                remainder = round((score/total_reps-full_rounds)*total_reps)
                x=[c*full_rounds for c in reps]
                f=[]
                for i in reps:
                    if remainder-i>0:
                        f.append(i)
                    else:
                        f.append(remainder)
                        check = len(reps)-len(f)
                        #print("c")
                        if check !=0:
                            #print("check")
                            for i in range(check):
                                f.append(0)
                        break
                    remainder=remainder-i
                    
                final_vals = np.array(x)+np.array(f)
                
                final_val_dict = {}
                for i in range(0,len(movements)):
                    if movements[i] not in final_val_dict:
                        final_val_dict[movements[i]]=0
                    final_val_dict[movements[i]]+=final_vals[i]
                final_vals=list(final_val_dict.values())
                for i in range(0,len(final_vals)):
                    final_dict[list(final_dict.keys())[i]].append(final_vals[i])
                final_dict['avg_time_per_round'].append('00:00.00')
    elif (df['type'].values[0]=="for_time"):
#print("hi")
        if workout in ['16.5','14.5','15.5']:
            time_domain=0
        else:
            time_domain = timedelta(minutes=int(df['time_cap'].values[0]))
        if workout in special:
            score_data['scoredisplay_'+str(workout_num)] = np.where(score_data['scoredisplay_'+str(workout_num)]=="430",score_data['breakdown_'+str(workout_num)].apply(lambda s: s[s.find("(")+1:s.find(")")]),score_data['scoredisplay_'+str(workout_num)].apply(lambda x: x + " reps"))
        scores=score_data['scoredisplay_'+str(workout_num)].values
        #print(scores)
        scores_dict={}
        for i in range(0,len(movements)):
            if movements[i] != "rest":
                if movements[i] not in scores_dict:
                    scores_dict[movements[i]]=0
                scores_dict[movements[i]]+=reps[i]
        #final_dict = {}
        #final_dict['avg_time_per_round']=[]
        
        for score in scores:
            if "reps" not in score:
                score = timedelta(minutes=int(score[:score.find(":")]),seconds=int(score[score.find(":")+1:]))
                #final_dict['avg_time_per_round'].append(format_time(score/rounds))
                #final_dict =d
                final_vals = list(scores_dict.values())
                #print(final_vals)
                #print(final_dict.keys())
                for i in range(0,len(final_vals)):
                    final_dict[list(final_dict.keys())[i]].append(final_vals[i])
            else:
                tracker={}
                #print(score)
                score = int(score[:score.find(" ")])
                for i in range(0,len(movements)):
                    if movements[i] != "rest":
                        if score - reps[i]>0:
                            if movements[i] not in tracker:
                                tracker[movements[i]]=0
                            tracker[movements[i]] += reps[i]
                            score -= reps[i]
                            
                        else:
                            #print("here")
                            for j in movements:
                                if j!="rest":
                                    if j not in tracker:
                                        tracker[j]=0
                            if movements[i] not in tracker:
                                tracker[movements[i]]=0
                            tracker[movements[i]]+=score
                            break
                    #print(score)
                    #print(tracker)
                final_vals = list(tracker.values())
                for i in range(0,len(final_vals)):
                    final_dict[list(final_dict.keys())[i]].append(final_vals[i])
                        
        total_reps=0
    elif (df['type'].values[0]=="for_load"):
        total_reps=0
        time_domain=0

    
            
    return final_dict,movements,total_reps,time_domain,d