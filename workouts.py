
import streamlit as st
import psycopg2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import timedelta
import plotly.graph_objects as go
import copy
import plotly.express as px
import plotly.figure_factory as ff
from total_reps import create_conn,load_data,load_result_data,format_time,calc_total_reps,calc_table_height,flatten_list,gen_table_colors


def app():

    st.header("Workout Analysis")

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    special=['16.2']

    df_move = load_data('movements')
    df_rep = load_data('rep_rounds')
    df_mbw = load_data('movements_by_workout')
    df_workout_desc = load_data('workout_desc')
    df_table = load_data("movements_label")
    df_table = df_table.fillna('')
    df_weight=load_data('weight')

    workout = st.selectbox(label="Workout",options=df_mbw['workout'])

    year = df_mbw[df_mbw['workout']==workout]['year'].values[0]

    gender = st.selectbox(label="Gender",options=["Men","Women"])

    bucket = st.text_input(label="Top xxx athletes",value="50")

    if "a" in workout:
        workout_num = workout[workout.find(".")+1:]
    else:
        workout_num = int(workout[workout.find(".")+1:])

    workout_text = df_workout_desc[df_workout_desc['workout']==workout]['workout_desc'].values[0]
    workout_text=workout_text.replace(r'\n','\n')

    score_data = load_result_data(str.lower(gender),int(year),workout_num,int(bucket))    
    final_dict,movements,total_reps,time_domain,d = calc_total_reps(workout,score_data,df_rep,workout_num,gender,special)
    st.subheader("Workout Description")
    st.markdown(workout_text)
    #st.text(final_dict)
    #st.text(d.keys())
    label_exceptions={'squat_clean': 'Squat Clean','snatch': 'Snatch','deadlift': 'Deadlift','clean_and_jerk': 'Clean and Jerk','squat_snatch': 'Squat Snatch'}
    movements_labeled =[]
    for m in list(d.keys()):
        st.text(d.keys())
        if m != "rest":
            if m[:-2] not in label_exceptions.keys():
                movements_labeled.append(df_move[df_move['movement']==m]['label'].values[0])
            else:
                #st.text([v for v in list(d.keys()) if m[:-2] in v])
                if len([v for v in list(d.keys()) if m[:-2] in v]) >1:
                    l = label_exceptions[m[:-2]]+" "+str(m[-1:])
                else:
                    l= label_exceptions[m[:-2]]
                movements_labeled.append(l)
                
    
    if (df_rep[df_rep['workout']==workout]['type'].values[0]=="AMRAP") & (pd.isnull(df_rep[df_rep['workout']==workout]['movement_1_rep_addition'].values[0])):
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df['breakdown_'+str(workout_num)]=final_df["breakdown_"+str(workout_num)].apply(lambda x: x.replace(r'\n','\n') if not pd.isnull(x) else x)
        final_df['raw_score'] = final_df['scoredisplay_'+str(workout_num)].apply(lambda score: int(score[:score.find(" ")]) if "reps" in score else int(score))
        final_df_copy=final_df.drop(columns=["scoredisplay_"+str(workout_num),"rounds","reps"])
        avg_df = round(final_df_copy.mean(axis=0))
        avg_df['raw_score']=round(avg_df['raw_score'])
        avg_df['rounds']=round(avg_df['raw_score']//total_reps)
        avg_df['reps']=(round(avg_df['raw_score']/total_reps)-avg_df['rounds'])*total_reps
        x=str(time_domain/avg_df['raw_score']*total_reps)
        x=':'.join(x.split(':')[1:])
        avg_df['time_per_round']=x
    elif (df_rep[df_rep['workout']==workout]['type'].values[0]=="AMRAP") or (df_rep[df_rep['workout']==workout]['type'].values[0]=="to_failure"):
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df['breakdown_'+str(workout_num)]=final_df["breakdown_"+str(workout_num)].apply(lambda x: x.replace(r'\n','\n') if not pd.isnull(x) else x)
        final_df['raw_score'] = final_df['scoredisplay_'+str(workout_num)].apply(lambda score: int(score[:score.find(" ")]) if "reps" in score else int(score))
        final_df=final_df.drop(columns=["scoredisplay_"+str(workout_num)])
        avg_df = round(final_df.mean(axis=0))
    elif df_rep[df_rep['workout']==workout]['type'].values[0]=="for_load":
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df=final_df[['competitorname','scoredisplay_'+str(workout_num)]]
        final_df['raw_score'] = final_df['scoredisplay_'+str(workout_num)].apply(lambda score: int(score[:score.find(" l")]) if "l" in score else int(score))
        avg_df = round(final_df.mean(axis=0))
        if "scoredisplay_"+str(workout_num) in avg_df.index:
            avg_df = avg_df.drop(index=["scoredisplay_"+str(workout_num)])
        #avg_df = avg_df.drop(columns=[movement_col])
        avg_df['raw_score']=str(int(avg_df['raw_score']))+ " lbs"
        final_df=final_df.drop(columns=['raw_score'])
    else:
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df['breakdown_'+str(workout_num)]=final_df["breakdown_"+str(workout_num)].apply(lambda x: x.replace(r'\n','\n') if not pd.isnull(x) else x)
        avg_df = round(final_df.mean(axis=0))
        avg_df['finishers']=len(final_df)-len(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains("reps")])
        avg_df['average_finish_time']=format_time(np.mean(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains(":")]['scoredisplay_'+str(workout_num)].apply(lambda score: timedelta(minutes=int(score[:score.find(":")]),seconds=int(score[score.find(":")+1:])))))
    #final_df['rank_'+str(workout_num)],final_df.competitorname,
    #final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)],final_df.wall_walk,final_df.double_under
    col_names = ['Rank','Athlete Name','Score','Score Detail']
    col_names.extend(movements_labeled)
    vals = [final_df['rank_'+str(workout_num)],final_df.competitorname,final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)]]
    vals_to_add = [final_df[m] for m in list(d.keys()) if m != 'rest']
    vals.extend(vals_to_add)
    final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)]
    table_colors=gen_table_colors(final_df,rowEvenColor,rowOddColor)
    fig_final = go.Figure(data=[go.Table(columnwidth=[1,1.5,1,1,1],header=dict(values=col_names,
    fill_color=headerColor,
    font=dict(color='white', size=18),
    line_color='darkslategray',),
    cells=dict(values=vals,
        line_color='darkslategray',
        fill_color = [table_colors*6],
        font = dict(size = 16),
        align = ['center','left',"center"],
        height=30))],)
    fig_final.update_layout(margin=dict(l=10,r=10, b=10,t=10),width=1200)

    fig_average = go.Figure(data=[go.Table(columnwidth=[1,1.5,1,1,1],header=dict(values=["Hi"],
    fill_color=headerColor,
    font=dict(color='white', size=18),
    line_color='darkslategray',),
    cells=dict(values=avg_df.values,
        line_color='darkslategray',
        fill_color = [table_colors*2],
        font = dict(size = 16),
        align = ['center','left',"center"],
        height=30))],)
    fig_average.update_layout(margin=dict(l=10,r=10, b=10,t=10))



    st.plotly_chart(fig_final)  
    st.plotly_chart(fig_average)
    #st.dataframe(final_df)
    st.dataframe(avg_df)

    #df_2020['reps']=df_2020['scoredisplay_2'].apply(calc_total_reps)
    final_dict,movements,total_reps,time_domain,d = calc_total_reps(workout,score_data,df_rep,workout_num,gender,special)