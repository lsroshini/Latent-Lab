"""Local research application; reconstruction remains the default page."""
import streamlit as st

st.navigation([
    st.Page('reconstruction_explorer.py',title='Reconstruction explorer',default=True),
    st.Page('supervised_comparison.py',title='Supervised AE comparison',url_path='supervised-comparison'),
]).run()
