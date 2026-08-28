import streamlit as st


def load_css():

    st.markdown(
        """
        <style>
        /* ==========================================================
   DASHBOARD SECTION
========================================================== */

.dashboard-section{

    background:white;

    padding:22px;

    border-radius:18px;

    margin-bottom:18px;

    border:1px solid #E3EAF2;

    box-shadow:0 4px 14px rgba(0,0,0,.06);

}

.section-title{

    font-size:24px;

    font-weight:700;

    color:#0B5394;

    margin-bottom:18px;

    border-bottom:2px solid #EEF3F9;

    padding-bottom:10px;

}
/* =============================================
   METRICS
============================================= */

div[data-testid="metric-container"]{

    border-radius:14px;

    border:1px solid #E7EDF5;

    background:white;

    padding:14px;

    box-shadow:0 2px 8px rgba(0,0,0,.05);

}

div[data-testid="metric-container"] label{

    font-size:15px;

    font-weight:600;

}

div[data-testid="stProgress"]{

    margin-top:15px;

}
/* ===========================================
   DATAFRAME
=========================================== */

[data-testid="stDataFrame"]{

    border-radius:16px;

    overflow:hidden;

    border:1px solid #E7EDF5;

}

/* ===========================================
   PLOTLY
=========================================== */

.js-plotly-plot{

    border-radius:16px;

}

/* ===========================================
   INFO / SUCCESS
=========================================== */

div[data-testid="stAlert"]{

    border-radius:12px;

}

/* ===========================================
   REMOVE EMPTY SPACE
=========================================== */

.block-container{

    padding-top:1rem;

    padding-bottom:1rem;

}

hr{

    margin-top:12px;

    margin-bottom:12px;

}
        </style>
        """,
        unsafe_allow_html=True,
    )