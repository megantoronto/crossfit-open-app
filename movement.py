import streamlit as st
import psycopg2
#import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
from datetime import timedelta
import plotly.graph_objects as go
#import copy
import plotly.express as px
import plotly.figure_factory as ff
from total_reps import create_conn,load_data,load_result_data,format_time,calc_total_reps,calc_table_height,flatten_list,gen_table_colors


st.set_page_config(layout='wide')


def app():
    

    st.title('Movement Analysis')
    special = ['16.2']

    df_move = load_data('movements')
    df_rep = load_data('rep_rounds')
    df_mbw = load_data('movements_by_workout')
    df_workout_desc = load_data('workout_desc')
    df_table = load_data("movements_label")
    df_table = df_table.fillna('')
    df_weight=load_data('weight')

    
    
    layout=calc_table_height(df_table)-100


    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    table_colors=gen_table_colors(df_table,rowEvenColor,rowOddColor)

    for i in range(len(df_table)):
        if i % 2==0:
            table_colors.append(rowEvenColor)
        else:
            table_colors.append(rowOddColor)

    #st.dataframe(df_move[['label','total_count']])
    fig = go.Figure(data=[go.Table(columnwidth=[5,1,1,1,1,1,1,1,1,1,1],header=dict(values=['Movement','Count','2011',
    '2012','2013','2014','2015','2016','2017','2018','2019','2020','2021'],
    fill_color=headerColor,
    font=dict(color='white', size=18),
    line_color='darkslategray',),
    cells=dict(values=[df_table.label,df_table.total_count,
        df_table.year_2011,
        df_table.year_2012,
        df_table.year_2013,
        df_table.year_2014,
        df_table.year_2015,
        df_table.year_2016,
        df_table.year_2017,
        df_table.year_2018,
        df_table.year_2019,
        df_table.year_2020,
        df_table.year_2021],
        line_color='darkslategray',
        fill_color = [table_colors*13],
        font = dict(size = 16),
        align = ['left',"center"],
        height=30))],
        layout=dict(height=layout),
        
        )
    fig.update_layout(margin=dict(l=10,r=10, b=10,t=10),width=1200)

    #fig=ff.create_table(df_table[['label','total_count','year_2011','year_2012','year_2013','year_2014',
    #'year_2015','year_2016','year_2017','year_2018','year_2019','year_2020','year_2021']],)
    #fig.update_layout(width=1000)
    st.plotly_chart(fig)
    st.subheader("Select Movement:")
    movement = st.selectbox(label="Movement",options=df_move['label'])
    movement_col = df_move[df_move['label']==movement]['movement'].values[0]

    d=df_mbw[df_mbw[str(movement_col)]==1]
    d=d.dropna(axis=1,how="all")
    all_movements=[c for c in list(d.columns) if c in list(df_move['movement'])]

    d['combo']=''
    master_list=[]
    other_list=[]
    for index,row in d.iterrows():
        combo=[]
        #print(row['workout'])
        for m in all_movements:
            if m != movement_col:
            #print(row[m])
                if row[m]==1:
                    label=df_move[df_move['movement']==m]['label'].values[0]
                    combo.append(label)
        #x=copy.copy(combo)
        #d['combo']=", ".join(x)
        master_list.append(", ".join(combo))
        other_list.append(combo)
    f=pd.DataFrame(master_list,columns=['combinations'])
    d=d.reset_index(drop=True)
    d=pd.concat([d,f],axis=1)
    s=flatten_list(other_list)
    combo_dict={}
    for i in s:
        if i not in combo_dict:
            combo_dict[i]=[0]
        combo_dict[i][0]+=1
    combo_dict=dict(sorted(combo_dict.items(), key=lambda item: item[1],reverse=True))

    """ fig, ax = plt.subplots(1, 1, tight_layout=True)
    langs = list(combo_dict.keys())
    students = combo_dict.values()
    ax.barh(langs,students,color='#B64926')
    #ax = plt.gca()
    for i, v in enumerate(students):
        ax.text(v + .05, i - .5, str(v))
    #ax.tick_params(axis='x', labelrotation = 45)
    plt.title("Movements Paired With " + movement)
    plt.xlabel("Number of Workouts Paired With " + movement)
    plt.ylabel("Movement")
    fig.set_size_inches(10, 5) """


    df_combo = pd.DataFrame(combo_dict).transpose().reset_index()
    df_combo.columns=['Movement','Count']
    fig=px.bar(df_combo,x='Count',y="Movement",orientation='h',text=df_combo['Count'],title='Movements Paired With '+movement,
    width=900,height=700,labels=dict(Count="Number of Times Paired With"))
    
    

    fig_combo =  go.Figure(data=[go.Table(columnwidth=[1,1,6],header=dict(values=['Year','Workout','Paired Movements'],
    fill_color=headerColor,
    font=dict(color='white', size=18),
    line_color='darkslategray',),
    cells=dict(values=[d.year,d.workout,
        d.combinations],
        line_color='darkslategray',
        fill_color = [table_colors*3],
        font = dict(size = 16),
        align = ['left',"center"],
        height=30))],layout=dict(height=calc_table_height(d)-100))
    fig_combo.update_layout(margin=dict(l=10,r=10, b=10,t=10),width=600)
    
    st.plotly_chart(fig_combo)
    movements_exclude=["deadlift_clean_hang_clean_overhead_complex"]
    if movement_col not in movements_exclude:
        st.plotly_chart(fig)
    weighted_movements=['thruster','clean','power_clean','squat_clean_and_jerk','push_press','front_squat','shoulder_to_overhead','overhead_walking_lunge',
                    'squat_snatch','single_dumbbell_box_step_up','single_arm_dumbbell_overhead_walking_lunge',
                    'single_arm_dumbbell_hang_clean_and_jerk',
                    'dumbbell_front_squat',
                    'dumbbell_front_rack_walking_lunge',
                    'dumbbell_power_clean',
                   'ground_to_overhead',
                   'dumbbell_thruster',
                   'squat_clean',
                   'power_snatch',
                   'single_arm_dumbbell_snatch',
                   'clean_and_jerk',
                   'overhead_squat',
                   'snatch',
                   'power_snatch',
                   'wall_ball','deadlift']
    if movement_col in weighted_movements:
        if "dumbbell" in movement_col:
            movement_col="dumbell"
        men_cols = list([i for i in list(df_weight.columns) if "men_"+movement_col+"_weight" in i and "women" not in i])
        women_cols = list([i for i in list(df_weight.columns) if "women_"+movement_col+"_weight" in i])
        num=len(women_cols)
        women_cols.append('year')
        women_cols.append('workout')
        men_cols.append('year')
        men_cols.append('workout')
        #df_women=df_weight[['year','workout','women_'+movement_col+'_weight']].dropna(how="any")
        df_women=df_weight[women_cols]
        df_women=df_women.dropna(subset=df_women.columns.difference(['year','workout']),how='all')
        df_women['gender']="F"
        if num >1:
            for i in range(num):
                df_women[movement_col+"_weight_"+str(i+1)]=df_women['women_'+movement_col+'_weight_'+str(i+1)]
        else:
            df_women[movement_col+"_weight"]=df_women['women_'+movement_col+'_weight']
        #df_men=df_weight[['year','workout','men_'+movement_col+'_weight']].dropna(how="any")
        df_men=df_weight[men_cols]
        df_men=df_men.dropna(subset=df_men.columns.difference(['year','workout']),how='all')
        df_men['gender']="M"
        if num >1:
            for i in range(num):
                df_men[movement_col+"_weight_"+str(i+1)]=df_men['men_'+movement_col+'_weight_'+str(i+1)]
        else:
            df_men[movement_col+"_weight"]=df_men['men_'+movement_col+'_weight']
        df=pd.concat([df_women,df_men])
        if movement_col in ['deadlift','snatch','squat_clean','squat_snatch','clean_and_jerk']:
            col1, col2 = st.columns(2)
            df_melt = pd.melt(df_women).dropna(how="any")
            women = df_melt[df_melt['variable'].str.contains("women")]
            s=[]
            test=women.drop_duplicates()
            for index,row in test.iterrows():
                s.append(df_women[df_women[row['variable']]==row['value']]['year'].values)
            years=[]
            counter=0
            for i in s:
                if len(i)==1:
                    years.append(i[0])
                    counter+=1
                if len(i)>1 and counter<len(women):
                    for j in i:
                        years.append(j)
                        counter+=1
            women['year']=years
            fig_2 = px.line(women, x="year", y="value",color="variable", title="Women "+movement+ " Weight 2011-2021")
            fig_2.update_xaxes(type='date',  tickformat="%Y")
            fig_2.update_traces(mode="markers+lines", hovertemplate=None)
            fig_2.update_layout(hovermode="x")

            col1.plotly_chart(fig_2)
            df_melt = pd.melt(df_men).dropna(how="any")
            men = df_melt[df_melt['variable'].str.contains("men")]
            s=[]
            test=men.drop_duplicates()
            for index,row in test.iterrows():
                s.append(df_men[df_men[row['variable']]==row['value']]['year'].values)
            years_men=[]
            counter=0
            for i in s:
                if len(i)==1:
                    years_men.append(i[0])
                    counter+=1
                if len(i)>1 and counter<len(men):
                    for j in i:
                        years_men.append(j)
                        counter+=1
            men['year']=years_men
            fig_3 = px.line(men, x="year", y="value",color="variable", title="Men "+movement+" Weight 2011-2021")
            fig_3.update_xaxes(type='date',  tickformat="%Y")
            fig_3.update_traces(mode="markers+lines", hovertemplate=None)
            fig_3.update_layout(hovermode="x")
            col2.plotly_chart(fig_3)
        else:
            fig_2 = px.line(df, x="year", y=movement_col+"_weight",color="gender", title=movement+ " Weight 2011-2021",width=800,height=600)
            fig_2.update_xaxes(type='date',  tickformat="%Y")
            fig_2.update_traces(mode="markers+lines", hovertemplate=None)
            fig_2.update_layout(hovermode="x")
            st.plotly_chart(fig_2)

    final_avgs=[]
    gender=st.selectbox(label="Gender",options=["Men","Women"])
    order = st.selectbox(label="Rank Type",options=["Workout Rank","Overall Rank"])
    workouts=d['workout']
    buckets = [50,500,10000]
    for workout in workouts:
        if "a" in workout:
            workout_num = workout[workout.find(".")+1:]
        else:
            workout_num = int(workout[workout.find(".")+1:])
        
        year="20"+workout[:workout.find(".")]
        score_data = load_result_data(gender.lower(),year,workout_num,max(buckets),order)
        final_dict,_,_,_,_=calc_total_reps(workout,score_data,df_rep,workout_num,gender.lower(),special)
        avg_list=[]
        for b in buckets:
            avg_list.append(np.mean(list(final_dict[movement_col])[:int(b)]))
        final_avgs.append(avg_list)

    df=pd.DataFrame(final_avgs)
    df.columns=['Top 50','Top 500','Top 10,000']
    df=round(df,1)
    df['Workout']=workouts
    df=df[['Workout','Top 50','Top 500','Top 10,000']]

    fig=go.Figure(data=[go.Table(header=dict(values=['Workout','Top 50','Top 500',
        'Top 10,000'],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[df['Workout'],df['Top 50'],
            df['Top 500'],
            df['Top 10,000']],
            line_color='darkslategray',
            fill_color = [table_colors*4],
            font = dict(size = 16),
            align = ["center"],
            height=30))],
            layout=dict(height=calc_table_height(df)-100)
            )
    fig.update_layout(margin=dict(l=10,r=10, b=10,t=10),width=600)
    if order=="Workout Rank":
        st.markdown("__Average Reps "+movement+" Per Top Workout Rank__")
    else:
        st.write("__Average Reps "+movement+" Per Top Overall Rank__")
    st.plotly_chart(fig)

    avg_df=pd.DataFrame(df[['Top 50','Top 500','Top 10,000']].mean()).reset_index()
    avg_df.columns = ['Group','Average Total Reps']
    avg_df = round(avg_df, 1)

    fig_2 = go.Figure(data=[go.Table(header=dict(values=['Group','Average Total Reps'],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[avg_df['Group'],avg_df['Average Total Reps']],
            line_color='darkslategray',
            fill_color = [table_colors*4],
            font = dict(size = 16),
            align = ["center"],
            height=30))],
            )
    fig_2.update_layout(margin=dict(l=10,r=10, b=10,t=10))
    if order=="Workout Rank":
        st.write("__Total Average Reps "+movement+" Per Top Workout Rank Across All Workouts__")
    else:
        st.write("__Total Average Reps "+movement+" Per Top Overall Rank Across All Workouts__")

    st.plotly_chart(fig_2)

            
    # y=d['type'].value_counts()
    # fig, ax = plt.subplots()
    # ax.set_ylabel('# of Workouts')
    # ax.set_title('Workout Types')
    # bars = plt.bar(y.index,height=y)
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x()+.3, yval + .1, "{:,}".format(yval))
    #st.pyplot(fig)

    
