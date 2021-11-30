import streamlit as st
from multiapp import MultiApp
import main, movement, workouts, introduction # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Introduction",introduction.app)
app.add_app("Home", main.app)
#app.add_app("Data Stats", main2.app)
app.add_app("Movement",movement.app)
app.add_app("Workouts",workouts.app)

# The main app
app.run()