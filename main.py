import streamlit as st
import psycopg2
#import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import plotly.express as px
import plotly.graph_objects as go
from total_reps import create_conn,gen_table_colors


def app():
    st.title('Demographics')



    #@st.cache()
    def load_data(table):
        conn = create_conn()
        query = '''SELECT * FROM ''' + table
        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def load_year_data(year,population):
        conn = create_conn()
        query = '''SELECT * FROM open_'''+str(year)+'''_women ORDER BY overallrank LIMIT ''' + str(population)
        #query = '''SELECT competitorname,age FROM open_'''+str(year)+'''_women LIMIT 50000'''
        df_women = pd.read_sql(query, conn)
        df_women['gender']="F"
        query = '''SELECT * FROM open_'''+str(year)+'''_men ORDER BY overallrank LIMIT ''' + str(population)
        df_men = pd.read_sql(query, conn)
        df_men['gender']="M"
        data=pd.concat([df_women,df_men])
        conn.close()
        return data
    
    #@st.cache(allow_output_mutation=True,hash_funcs={"_thread.RLock": lambda _: None})
    def load_all_age_data(years,population="ignore"):
        conn=create_conn()
        final_data=pd.DataFrame()
        for year in years:
            if population == "ignore":
                query = '''SELECT age FROM open_'''+str(year)+'''_women ORDER BY overallrank '''
            else:
                query = '''SELECT age FROM open_'''+str(year)+'''_women ORDER BY overallrank LIMIT '''+str(population)
            #query = '''SELECT competitorname,age FROM open_'''+str(year)+'''_women LIMIT 50000'''
            df_women = pd.read_sql(query, conn)
            df_women['gender']="F"
            df_women['year']=year
            if population=="ignore":
                query = '''SELECT age FROM open_'''+str(year)+'''_men ORDER BY overallrank '''
            else:
                query = '''SELECT age FROM open_'''+str(year)+'''_men ORDER BY overallrank LIMIT '''+str(population)
            df_men = pd.read_sql(query, conn)
            df_men['gender']="M"
            df_men['year']=year
            data=pd.concat([df_women,df_men])
            final_data=pd.concat([final_data,data])
        return final_data

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    
    df_tot = load_data("total_competitors")
    table_colors=gen_table_colors(df_tot,rowEvenColor,rowOddColor)

    st.subheader('CrossFit Open Participation (Age 16-54)')
    check=st.selectbox("View",options=["Graph","Table"])

    if check=="Table":
        df_tot['men']=df_tot['men'].map(u"{:,}".format)
        df_tot['women']=df_tot['women'].map(u"{:,}".format)
        df_tot['total']=df_tot['total'].map(u"{:,}".format)
        fig=go.Figure(data=[go.Table(header=dict(values=['Year','Men','Women',
        'Total'],
        fill_color=headerColor,
        font=dict(color='white', size=18),
        line_color='darkslategray',),
        cells=dict(values=[df_tot.year,df_tot.men,
            df_tot.women,
            df_tot.total],
            line_color='darkslategray',
            fill_color = [table_colors*4],
            font = dict(size = 16),
            align = ["center"],
            height=30))],
            )
        fig.update_layout(margin=dict(l=10,r=10, b=10,t=10),width=600)
        st.plotly_chart(fig)
    else:
        
        df_tot_men=df_tot[['year','men']]
        df_tot_men['gender']="M"
        df_tot_men.columns=['year','count','gender']
        df_tot_women = df_tot[['year','women']]
        df_tot_women['gender']="F"
        df_tot_women.columns=['year','count','gender']
        df_tot_tot = df_tot[['year','total']]
        df_tot_tot['gender']="Total"
        df_tot_tot.columns=['year','count','gender']
        df_stack = pd.concat([df_tot_men,df_tot_women,df_tot_tot])
        df_stack['year']=df_stack['year'].astype('int')
        df_stack.columns=['Year','Count','Gender']
        fig = px.line(df_stack, x="Year", y="Count",color="Gender", markers=True,width=800,height=600)
        fig.update_yaxes(tickformat=",")
        fig.update_xaxes(dtick=1)

        st.plotly_chart(fig)

    st.subheader('Age Category Breakdown')
    gender=st.selectbox(label="Gender",options=["Men","Women"])

    years=["2021","2020","2019","2018","2017","2016","2015","2014","2013","2012","2011"]
    df_all = load_all_age_data(years)
    ages=df_all['age']
    bins=[0,10,17,24,30,34,39,44,49,54,np.inf]
    labels=["Unknown","16-17","18-24","25-30","31-34","35-39","40-44","45-49","50-54","Unknown"]
    groups = pd.cut(ages, bins=bins, labels=labels,ordered=False)
    df=pd.concat((ages, groups,df_all['gender'],df_all['year']), axis=1)
    df.columns=['Age','Age Category','Gender','Year']
    if gender=="Women":
        df=df[df["Gender"]=="F"]
    else:
        df=df[df["Gender"]=="M"]


    df=pd.DataFrame(df.groupby(['Age Category','Year']).size()).reset_index()
    df.columns=['Age Category','Year','Count']
    fig = px.line(df, x="Year", y="Count",color="Age Category", markers=True,title="CrossFit Open Participation " + gender +" (Ages 16-54)",
    width=800,height=600)
    fig.update_yaxes(tickformat=",")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig)

    st.markdown("__Strat Age Data by Year and Athlete Rank__")

    year = st.selectbox(label="Year",options=["2021","2020","2019","2018","2017","2016","2015","2014","2013","2012","2011","All Years - Women","All Years - Men"])
    population = st.selectbox(label="Select # of Athletes",options=[50,100,500,1000,5000,10000])

    if "All" in year:
        df_all = load_all_age_data(years,50)
        ages=df_all['age']
        bins=[0,10,17,24,30,34,39,44,49,54,np.inf]
        labels=["Unknown","16-17","18-24","25-30","31-34","35-39","40-44","45-49","50-54","Unknown"]
        groups = pd.cut(ages, bins=bins, labels=labels,ordered=False)
        df=pd.concat((ages, groups,df_all['gender'],df_all['year']), axis=1)
        df.columns=['Age','Age Category','Gender','Year']
        if "Women" in year:
            df=df[df["Gender"]=="F"]
        else:
            df=df[df["Gender"]=="M"]
        df_stack=pd.DataFrame(df.groupby(['Year','Age Category']).size()).reset_index()
        df_stack['Percentage']=df.groupby(['Year','Age Category']).size().groupby(level=0).apply(lambda 
        x:100 * x/float(x.sum())).values
        df_stack.columns=['Year','Age Category','Count',"Percentage"]
        fig = px.bar(df_stack, x="Year", y="Percentage",color="Age Category",title="CrossFit Open Participation (Ages 16-54)",
            text=df_stack['Percentage'].apply(lambda x: '{0:1.0f}%'.format(x)),custom_data=[df_stack['Age Category'],df_stack['Count']],
            width=800,height=600)
        #fig.update_yaxes(tickformat="%")
        fig.update_traces(hovertemplate='Gender: %{x}<br>Age Category: %{customdata[0]} <br>Percentage: %{text} <br>Count: %{customdata[1]:,}')
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig)

    else:
        df_age = load_year_data(year,population)

        ages=df_age['age']
        bins=[0,10,17,24,30,34,39,44,49,54,np.inf]
        labels=["Unknown","16-17","18-24","25-30","31-34","35-39","40-44","45-49","50-54","Unknown"]
        groups = pd.cut(ages, bins=bins, labels=labels,ordered=False)
        df=pd.concat((ages, groups,df_age['gender']), axis=1)
        df.columns = ['Age','Age Category','Gender']
        df_stack=df.groupby(['Gender','Age Category']).size().reset_index()
        df_stack['Percentage']=df.groupby(['Gender',
                'Age Category']).size().groupby(level=0).apply(lambda 
                    x:100 * x/float(x.sum())).values
        df_stack.columns= ['Gender', 'Age Category','Count','Percentage']
            #df_stack['Percentage'] =  df_stack['Percentage'].map('{:,.2f}%'.format) 
        fig = px.bar(df_stack, x="Gender", y="Percentage", color="Age Category",
                        barmode='stack',text=df_stack['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
                        custom_data=[df_stack['Age Category'],df_stack['Count']],
                        title="Age Category Breakdown For Top " + str(population) + " Athletes",
                        width=1000,height=800)
            #category_orders={"Age Category": ["Unknown","16-17","18-24","25-30","31-34","35-39","40-44","45-49","50-54"],}
        fig.update_traces(hovertemplate='Gender: %{x}<br>Age Category: %{customdata[0]} <br>Percentage: %{text} <br>Count: %{customdata[1]:,}')
        st.plotly_chart(fig)

    st.subheader("Height Breakdown")

    year_h = st.selectbox(label="Year",options=["2021","2020","2019","2018","2017"])
    gender=st.selectbox(label="Gender",options=["Men","Women"],key="height_gen")
    population_h = st.selectbox(label="Select # of Athletes",options=[50,100,500,1000,5000,10000],key='height')

    df_h = load_year_data(year_h,population_h)
    df_h=df_h[['competitorname','gender','height']]
    df_h_not = df_h.dropna(how='any')
    df_h_not['height_old']=df_h_not['height']
    #df_h_not['height']=np.where(df_h_not['height']=="15700cm","157 cm",df_h_not['height'])
    df_h_not['height']=df_h_not['height'].apply(lambda x: str(round((int(x[:-3])/2.54))) if "cm" in x else x)
    df_h_not['height']=df_h_not['height'].apply(lambda x: str(round((int(x[:-3])))) if "in" in x else x)
    df_h_not['height']=df_h_not['height'].apply(lambda x: int(x[:x.find("'")])*12+int(x[x.find("'")+1:x.find('"')]) if "'" in x else x )
    df_h_not['height']=df_h_not['height'].astype('int')
    height_ranges=[55,95]
    df_h_not=df_h_not[(df_h_not['height']>=height_ranges[0]) & (df_h_not['height']<=height_ranges[1])]
    df_h_not['height_label']=df_h_not['height'].apply(lambda x: str(x//12)+"'"+str((x%12))+'''"''' )
    if gender=="Men":
        df=df_h_not[df_h_not["gender"]=="M"]
    else:
        df=df_h_not[df_h_not["gender"]=="F"]
    avg_height=round(np.mean(df['height']))
    avg_height=str(avg_height//12)+"'"+str((avg_height%12))+'''"'''
    df=pd.DataFrame(df.groupby(['height_label','height']).size()).reset_index()
    df.columns=['Height','height_num','Count']
    if population_h-np.sum(df['Count'])>0:
        df=pd.concat([df,pd.DataFrame(data={'Height':['Unknown'],'Count':[population_h-np.sum(df['Count'])]})])
    df=df.sort_values(by=['height_num'])

    fig=px.bar(df,x="Height",y='Count',text=df['Count'].map(u"{:,}".format),
    title="Height of " + str(year_h)+" Top " + str("{:,}".format(population_h)) + " "+gender,width=800,height=600)
    fig.add_annotation(text="Average Height: "+avg_height,
                xref="paper", yref="paper",
                x=1, y=1, showarrow=False,font=dict(size=16),)
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig)

    st.subheader("Weight Breakdown")

    year_w = st.selectbox(label="Year",options=["2021","2020","2019","2018","2017"],key="weight")
    gender_w=st.selectbox(label="Gender",options=["Men","Women"],key="weight_gen")
    population_w = st.selectbox(label="Select # of Athletes",options=[50,100,500,1000,5000,10000],key='weight_pop')


    df_w=load_year_data(year_w,population_w)
    df_w=df_w[['competitorname','gender','weight']]
    df_w['weight_old']=df_w['weight']
    df_w['weight']=df_w['weight'].fillna("0")#dropna(how='any')
    df_w['weight']=df_w['weight'].apply(lambda x: str(round(int(x[:-3])*2.20462)) if "kg" in x else x)
    df_w['weight']=df_w['weight'].apply(lambda x: str(x[:-3]) if "lb" in x else x)
    df_w['weight']=df_w['weight'].astype('int')

    weights=df_w['weight']
    bins=[i for i in range(100,255,5)]
    bins.insert(0,0)
    bins.insert(len(bins),np.inf)
    labels=['Unknown',
    '100-105',
    '105-110',
    '110-115',
    '115-120',
    '120-125',
    '125-130',
    '130-135',
    '135-140',
    '140-145',
    '145-150',
    '150-155',
    '155-160',
    '160-165',
    '165-170',
    '170-175',
    '175-180',
    '180-185',
    '185-190',
    '190-195',
    '195-200',
    '200-205',
    '205-210',
    '210-215',
    '215-220',
    '220-225',
    '225-230',
    '230-235',
    '235-240',
    '240-245',
    '245-250',
    '250+']
    groups = pd.cut(weights, bins=bins, labels=labels,ordered=False)
    df=pd.concat((weights, groups,df_w['gender']), axis=1)
    df.columns=['Weight','Weight Category','gender']
    if gender_w=="Women":
        df=df[df['gender']=="F"]
    else:
        df=df[df['gender']=="M"]
    avg_weight=round(np.mean(df[df['Weight Category']!="Unknown"]['Weight']))
    df=pd.DataFrame(df.groupby(['Weight Category']).size()).reset_index()

    df.columns=['Weight Category','Count']
    #df.loc[~(df==df.loc[(df!=0).any(axis=1)])]
    df=df[df['Count']!=0]

    fig=px.bar(df,x="Weight Category",y='Count',text=df['Count'].map(u"{:,}".format),
    title="Weight (lbs) of " +str(year_w) +" Top "+str("{:,}".format(population_w))+" "+gender_w,
    width=800,height=600)
    fig.add_annotation(text="Average Weight: "+str(avg_weight)+"lbs",
                    xref="paper", yref="paper",
                    x=1, y=1, showarrow=False,font=dict(size=16),)
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig)