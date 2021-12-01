import streamlit as st
from multiapp import MultiApp
import introduction,main, movement, workouts, workout_performance  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Introduction",introduction.app)
app.add_app("Demographics", main.app)
app.add_app("Movements",movement.app)
app.add_app("Workouts",workouts.app)
app.add_app("Workout Performance",workout_performance.app)

# The main app
app.run()