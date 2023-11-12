""""
# TCREdit: Editing SARS-CoV-2 T-cell epitope-specific TCRB sequences
"""
import streamlit as st
import streamlit_datalist as datalist
import pandas as pd
import numpy as np
from tcredit.common import FileUtils
from tcredit.data import CN


def change_label_style(label, font_size='12px', font_color='black', font_family='sans-serif'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    st.components.v1.html(html)


st.title('TCREdit')
st.subheader('Editing SARS-CoV-2 T-cell epitope-specific TCR$\\beta$ sequences')
st.markdown('---')

dm = FileUtils.json_load('data.json')
df = pd.read_csv(dm['fn_sars2'])
epitopes = np.unique(df[CN.epitope_seq].values).tolist()
epitope_lens = df[CN.epitope_len].unique()
col1, col2 = st.columns([1, 3])
with col1:
    epitope = st.selectbox(label=r"$\textsf{\large Epitope:}$",
                           options=epitopes)

with col2:
    cdr3b = datalist.stDatalist(label=r"CDR3$\\beta$",
                                options=df[df[CN.epitope_seq] == epitope][CN.cdr3b_seq].unique().tolist())
