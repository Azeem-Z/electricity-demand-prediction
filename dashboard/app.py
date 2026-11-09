# Electricity demand prediction - dashboard
# run from the code/ folder:  streamlit run dashboard/app.py

import streamlit as st

st.set_page_config(page_title="Electricity Demand Prediction", page_icon="⚡", layout="wide")

pages = {
    "Main": [
        st.Page("views/predict.py", title="Predict demand", icon="⚡", default=True),
        st.Page("views/results.py", title="Results", icon="📊"),
        st.Page("views/patterns.py", title="Temporal patterns", icon="📈"),
    ],
    "About": [
        st.Page("views/about.py", title="About the project", icon="🏠"),
        # hidden page, not in the menu - open it directly with /terms in the browser
        st.Page("views/glossary.py", title="Terms explained", icon="📖", url_path="terms"),
    ],
}

# hide the link of the terms page in the menu
st.markdown("<style>a[data-testid='stSidebarNavLink'][href$='/terms'] {display: none;}</style>",
            unsafe_allow_html=True)

st.navigation(pages).run()
