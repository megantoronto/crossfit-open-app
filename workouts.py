import streamlit as st
from total_reps import load_data
import pandas as pd
import plotly.express as px
import numpy as np

def app():
    st.title("Workout Analysis")

    st.subheader('Workout Types')



    df_mbw=load_data("movements_by_workout")
    df=pd.DataFrame(df_mbw[['type','year']].groupby('type').size()).reset_index()
    df.columns=['Workout Type','Count']
    fig=px.pie(df,values='Count',names='Workout Type',title="Total Number of Workouts: 57",width=800,height=600)
    fig.update_traces(textinfo='value+percent')

    st.plotly_chart(fig)


    df=pd.DataFrame(df_mbw[['type','year']].groupby(['type','year']).size()).reset_index()
    df.columns=['Workout Type',"Year","Count"]

    fig=px.bar(df,x='Year',y='Count',color="Workout Type",text=df['Count'],title="Count of Workout Types by Year",
    width=800,height=600)
    st.plotly_chart(fig)

    st.subheader('Time Domains')
    df=df_mbw[df_mbw['type'].isin(['AMRAP','for_time','for_load'])]
    df['time']=np.where(pd.isnull(df['time_domain']),df['time_cap'],df['time_domain'])
    df=df[['type','time','year']]
    df.columns=['Workout Type','Time','Year']
    fig=px.box(df,x='Workout Type',y="Time",labels=dict(Time="Minutes"),title="Maximum Length of Workout by Workout Type",
    width=800,height=600)
    st.plotly_chart(fig)

    time=df['Time']
    bins=[0,7,14,19,20]
    labels=["0-7min","8-14min","15-19min","20+min"]
    groups = pd.cut(time, bins=bins, labels=labels,ordered=False)
    df=pd.concat((time, groups,df['Year']), axis=1)
    df.columns = ['Time','Time Category','Year']
    df_stack=df.groupby(['Year','Time Category']).size().reset_index()
    df_stack['Percentage']=df.groupby(['Year',
        'Time Category']).size().groupby(level=0).apply(lambda 
            x:100 * x/float(x.sum())).values
    df_stack.columns= ['Year', 'Time Category','Count','Percentage']
    #df_stack['Percentage'] =  df_stack['Percentage'].map('{:,.2f}%'.format) 

    fig = px.bar(df_stack, x="Year", y="Count", color="Time Category",
                barmode='stack',text=df_stack['Count'],custom_data=[df_stack['Time Category'],df_stack['Percentage']],
                title="Maximum Time Domain Categorization by Year for AMRAPs, For Time, and For Load Workouts",
                width=800,height=600)
    #category_orders={"Age Category": ["Unknown","16-17","18-24","25-30","31-34","35-39","40-44","45-49","50-54"],}
    fig.update_traces(hovertemplate='Year: %{x}<br>Time Category: %{customdata[0]} <br>Percentage: %{customdata[1]} <br>Count: %{text}')
    st.plotly_chart(fig)







