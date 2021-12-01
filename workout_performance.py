
import streamlit as st
import psycopg2
#import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
from datetime import timedelta
import plotly.graph_objects as go
import copy
import plotly.express as px
import plotly.figure_factory as ff
from total_reps import create_conn,load_data,load_result_data,format_time,calc_total_reps,calc_table_height,flatten_list,gen_table_colors


def app():

    st.title("Workout Performance")

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

    dropdown = df_mbw.sort_values(['year','workout'],ascending=[False,True])['workout']

    workout = st.selectbox(label="Workout",options=dropdown)

    year = df_mbw[df_mbw['workout']==workout]['year'].values[0]

    gender = st.selectbox(label="Gender",options=["Men","Women"])

    bucket = st.text_input(label="Select # of Athletes",value="50")

    order = st.selectbox(label="Rank Type",options=["Workout Rank","Overall Rank"])


    if "a" in workout:
        workout_num = workout[workout.find(".")+1:]
    else:
        workout_num = int(workout[workout.find(".")+1:])

    workout_text = df_workout_desc[df_workout_desc['workout']==workout]['workout_desc'].values[0]
    workout_text=workout_text.replace(r'\n','\n')

    score_data = load_result_data(str.lower(gender),int(year),workout_num,int(bucket),order=order)    
    final_dict,movements,total_reps,time_domain,d = calc_total_reps(workout,score_data,df_rep,workout_num,gender,special)
    st.subheader("Workout Description")
    st.markdown(workout_text)
    #st.text(final_dict)
    #st.text(d.keys())
    label_exceptions={'squat_clean': 'Squat Clean','snatch': 'Snatch','deadlift': 'Deadlift','clean_and_jerk': 'Clean and Jerk','squat_snatch': 'Squat Snatch'}
    movements_labeled =[]
    for m in list(d.keys()):
        #st.text(d.keys())
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
        
        if workout not in ("14.3","13.1","12.1","12.2","11.3"):
            col_names = ['Workout Rank','Athlete Name','Score','Score Detail','Rounds','Reps','Avg Time Per Round']
            vals = [final_df['rank_'+str(workout_num)],final_df.competitorname,final_df['scoredisplay_'+str(workout_num)],
        final_df['breakdown_'+str(workout_num)],final_df['rounds'],final_df['reps'],final_df['avg_time_per_round']]
            col_names.extend(movements_labeled)
            vals_to_add = [final_df[m] for m in list(d.keys()) if m != 'rest']
            vals.extend(vals_to_add)
        else:
            col_names = ['Workout Rank','Athlete Name','Score','Score Detail','Rounds','Reps']
            vals = [final_df['rank_'+str(workout_num)],final_df.competitorname,final_df['scoredisplay_'+str(workout_num)],
        final_df['breakdown_'+str(workout_num)],final_df['rounds'],final_df['reps']]
            if workout == '14.3':
                col_names.extend(movements_labeled[:-1])
                vals_to_add = [final_df[m] for m in list(d.keys())[:-1] if m != 'rest']
            else:
                col_names.extend(movements_labeled)
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
        
        
        final_df['raw_score'] = final_df['scoredisplay_'+str(workout_num)].apply(lambda score: int(score[:score.find(" ")]) if "reps" in score else int(score))
        final_df_copy=final_df.drop(columns=["rounds","reps"])
        avg_df = round(final_df_copy.mean(axis=0))
        avg_df['Score']=round(avg_df['raw_score'])
        avg_df['Rounds']=round(avg_df['Score']//total_reps)
        #st.text(total_reps)
        avg_df['Reps']=round(((avg_df['Score']/total_reps)-avg_df['Rounds'])*total_reps)
        x=str(time_domain/avg_df['Score']*total_reps)
        x=':'.join(x.split(':')[1:])
        if workout not in ("14.3","13.1","12.1","12.2","11.3"):
            avg_df['Time Per Round']=x
        if 'scoredisplay_'+str(workout_num) in avg_df.index:
            avg_df=avg_df.drop(index=['scoredisplay_'+str(workout_num)])
        avg_df=avg_df.drop(index=['rank_'+str(workout_num),'raw_score'])
        avg_df = pd.DataFrame(avg_df).reset_index()
        avg_df.columns=['Movement','Average Reps']

        fig_average = go.Figure(data=[go.Table(header=dict(values=["Movement","Average Reps"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[avg_df['Movement'],avg_df['Average Reps']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ['center'],
            height=30))],layout=dict(height=calc_table_height(avg_df)-150))
        fig_average.update_layout(margin=dict(l=10,r=10, b=10,t=10))

        st.write("__Results__")
        st.plotly_chart(fig_final)  
        st.write("__Average Rounds, Reps, & Time Per Round__")
        st.plotly_chart(fig_average)

    
    elif df_rep[df_rep['workout']==workout]['type'].values[0]=="for_load":
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        #final_df=final_df[['competitorname','scoredisplay_'+str(workout_num)]]
        final_df['raw_score'] = final_df['scoredisplay_'+str(workout_num)].apply(lambda score: int(score[:score.find(" l")]) if "l" in score else int(score))
        avg_df = round(final_df.mean(axis=0))
        if "scoredisplay_"+str(workout_num) in avg_df.index:
            avg_df = avg_df.drop(index=["scoredisplay_"+str(workout_num)])
        #avg_df = avg_df.drop(columns=[movement_col])
        avg_df[movements_labeled[0]]=str(int(avg_df['raw_score']))+ " lbs"
        avg_df=avg_df.drop(index=['rank_'+str(workout_num),movements[0],'raw_score'])
        avg_df=pd.DataFrame(avg_df).reset_index()
        avg_df.columns=['Movement','Weight']
        final_df=final_df.drop(columns=['raw_score'])

        col_names = ['Workout Rank','Athlete Name','Score']
        #col_names.extend(movements_labeled)
        vals = [final_df['rank_'+str(workout_num)],final_df.competitorname,final_df['scoredisplay_'+str(workout_num)]]
        final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)]
        table_colors=gen_table_colors(final_df,rowEvenColor,rowOddColor)
        fig_final = go.Figure(data=[go.Table(columnwidth=[1,1.5,1],header=dict(values=col_names,
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

        fig_average = go.Figure(data=[go.Table(columnwidth=[1.5,1],header=dict(values=["Movement","Average Weight"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[avg_df['Movement'],avg_df['Weight']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ['center'],
            height=30))],layout=dict(height=calc_table_height(avg_df)-150))
        fig_average.update_layout(margin=dict(l=10,r=10, b=10,t=10))

        st.write("__Results__")
        st.plotly_chart(fig_final) 
        st.write("__Average Weight Lifted__") 
        st.plotly_chart(fig_average)

    elif (df_rep[df_rep['workout']==workout]['type'].values[0]=="AMRAP") or (df_rep[df_rep['workout']==workout]['type'].values[0]=="to_failure"):
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df['breakdown_'+str(workout_num)]=final_df["breakdown_"+str(workout_num)].apply(lambda x: x.replace(r'\n','\n') if not pd.isnull(x) else x)
        col_names = ['Workout Rank','Athlete Name','Score','Score Detail']
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

        final_df['Score'] = final_df['scoredisplay_'+str(workout_num)].apply(lambda score: int(score[:score.find(" ")]) if "reps" in score else int(score))

        avg_df = round(pd.DataFrame(final_df.mean(axis=0)))
        if "scoredisplay_"+str(workout_num) in avg_df.index:
            avg_df = avg_df.drop(index=["scoredisplay_"+str(workout_num)])
        avg_df=avg_df.drop(index=['rank_'+str(workout_num)])
        avg_df=avg_df.reset_index()
        #st.dataframe(avg_df)
        avg_df.columns = ['Movement','Average Reps']
        #avg_df['Movement']=avg_df['Movement'].apply(lambda x: df_move[df_move['movement']==x]['label'].values[0])

        fig_average = go.Figure(data=[go.Table(header=dict(values=["Movement","Average Reps"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[avg_df['Movement'],avg_df['Average Reps']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ['center'],
            height=30))],layout=dict(height=calc_table_height(avg_df)-150))
        fig_average.update_layout(margin=dict(l=10,r=10, b=10,t=10))

        st.write("__Results__")
        st.plotly_chart(fig_final)  
        st.write("__Average Reps Completed__")
        st.plotly_chart(fig_average)

    elif (df_rep[df_rep['workout']==workout]['type'].values[0]=="for_time") & pd.notnull(df_rep[df_rep['workout']==workout]['rounds'].values[0]):
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df['breakdown_'+str(workout_num)]=final_df["breakdown_"+str(workout_num)].apply(lambda x: x.replace(r'\n','\n') if not pd.isnull(x) else x)
        #
        
    #final_df['rank_'+str(workout_num)],final_df.competitorname,
    #final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)],final_df.wall_walk,final_df.double_under
        col_names = ['Workout Rank','Athlete Name','Score','Score Detail','Avg Time Per Round']
        col_names.extend(movements_labeled)
        vals = [final_df['rank_'+str(workout_num)],final_df.competitorname,final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)],final_df['avg_time_per_round']]
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
        
        final_df['Total Reps']=final_df['scoredisplay_'+str(workout_num)].apply(lambda x: int(x[:x.find(" ")]) if "reps" in x else total_reps)
        #st.dataframe(final_df)
        final_df=final_df.drop(columns=['rank_'+str(workout_num)])
        avg_df = round(pd.DataFrame(final_df.mean(axis=0)))
        finish=pd.DataFrame()
        finish['Finishers']=[len(final_df)-len(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains("reps")])]
        finish['Average Finish Time']=[format_time(np.mean(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains(":")]['scoredisplay_'+str(workout_num)].apply(lambda score: timedelta(minutes=int(score[:score.find(":")]),seconds=int(score[score.find(":")+1:])))))]
        finish['Average Time Per Round']=[format_time(np.mean(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains(":")]['scoredisplay_'+str(workout_num)].apply(lambda score: timedelta(minutes=int(score[:score.find(":")]),seconds=int(score[score.find(":")+1:]))))/int(df_rep[df_rep['workout']==workout]['rounds'].values[0]))]
        avg_df=avg_df.reset_index()
        #st.dataframe(avg_df)
        avg_df.columns = ['Movement','Average Reps']
        #avg_df['Movement']=avg_df['Movement'].apply(lambda x: df_move[df_move['movement']==x]['label'].values[0])

        fig_average = go.Figure(data=[go.Table(header=dict(values=["Movement","Average Reps"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[avg_df['Movement'],avg_df['Average Reps']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ['center'],
            height=30))],layout=dict(height=calc_table_height(avg_df)-150))
        fig_average.update_layout(margin=dict(l=10,r=10, b=10,t=10))

        fig_finish = go.Figure(data=[go.Table(header=dict(values=["Finishers","Average Finish Time","Average Time Per Round"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[finish['Finishers'],finish['Average Finish Time'],finish['Average Time Per Round']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ["center"],
            height=30))],)
        fig_finish.update_layout(margin=dict(l=10,r=10, b=10,t=10))




        st.write("__Results__")
        st.plotly_chart(fig_final)  
        st.write("__Average Reps Completed__")
        st.plotly_chart(fig_average)
        st.write("__Finisher Stats__")
        st.plotly_chart(fig_finish)

    else:
        final_df = pd.concat([score_data,pd.DataFrame(final_dict)],axis=1)
        final_df['breakdown_'+str(workout_num)]=final_df["breakdown_"+str(workout_num)].apply(lambda x: x.replace(r'\n','\n') if not pd.isnull(x) else x)
        #
        
    #final_df['rank_'+str(workout_num)],final_df.competitorname,
    #final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)],final_df.wall_walk,final_df.double_under
        col_names = ['Workout Rank','Athlete Name','Score','Score Detail']
        col_names.extend(movements_labeled)
        vals = [final_df['rank_'+str(workout_num)],final_df.competitorname,final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)]]
        vals_to_add = [final_df[m] for m in list(d.keys()) if m != 'rest']
        vals.extend(vals_to_add)
        final_df['scoredisplay_'+str(workout_num)],final_df['breakdown_'+str(workout_num)]
        table_colors=gen_table_colors(final_df,rowEvenColor,rowOddColor)
        fig_final = go.Figure(data=[go.Table(columnwidth=[1,1,1,1.5,1],header=dict(values=col_names,
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
        
        final_df['Total Reps']=final_df['scoredisplay_'+str(workout_num)].apply(lambda x: int(x[:x.find(" ")]) if "reps" in x else total_reps)
        #st.dataframe(final_df)
        final_df=final_df.drop(columns=['rank_'+str(workout_num)])
        avg_df = round(pd.DataFrame(final_df.mean(axis=0)))
        finish=pd.DataFrame()
        finish['Finishers']=[len(final_df)-len(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains("reps")])]
        finish['Average Finish Time']=[format_time(np.mean(final_df[final_df['scoredisplay_'+str(workout_num)].str.contains(":")]['scoredisplay_'+str(workout_num)].apply(lambda score: timedelta(minutes=int(score[:score.find(":")]),seconds=int(score[score.find(":")+1:])))))]
        avg_df=avg_df.reset_index()
        #st.dataframe(avg_df)
        avg_df.columns = ['Movement','Average Reps']
        #avg_df['Movement']=avg_df['Movement'].apply(lambda x: df_move[df_move['movement']==x]['label'].values[0])

        fig_average = go.Figure(data=[go.Table(header=dict(values=["Movement","Average Reps"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[avg_df['Movement'],avg_df['Average Reps']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ['center'],
            height=30))],layout=dict(height=calc_table_height(avg_df)-150))
        fig_average.update_layout(margin=dict(l=10,r=10, b=10,t=10))

        fig_finish = go.Figure(data=[go.Table(header=dict(values=["Finishers","Average Finish Time"],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[finish['Finishers'],finish['Average Finish Time']],
            line_color='darkslategray',
            fill_color = [table_colors*2],
            font = dict(size = 16),
            align = ["center"],
            height=30))],)
        fig_finish.update_layout(margin=dict(l=10,r=10, b=10,t=10))




        st.write("__Results__")
        st.plotly_chart(fig_final)  
        st.write("__Average Reps Completed__")
        st.plotly_chart(fig_average)
        st.write("__Finisher Stats__")
        st.plotly_chart(fig_finish)
    #st.dataframe(final_df)
        #st.dataframe(avg_df)
        #st.dataframe(finish)

    #df_2020['reps']=df_2020['scoredisplay_2'].apply(calc_total_reps)
    #final_dict,movements,total_reps,time_domain,d = calc_total_reps(workout,score_data,df_rep,workout_num,gender,special)