import streamlit as st

main_page = st.Page("main.py", title="Predictive Maintenance", icon=":material/cyclone:")
pg = st.navigation([main_page])
pg.run()