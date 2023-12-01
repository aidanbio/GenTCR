import gradio as gr
import numpy as np
import pandas as pd
from gentcr.data import CN, EpitopeTargetDataset

from gentcr.common import FileUtils

dm = FileUtils.json_load('data.json')
df = EpitopeTargetDataset.from_key(dm['data_key']).df
epitopes = df[CN.epitope_seq].unique().tolist()
sel_epitope = epitopes[0]

with gr.Blocks('GenTCR') as demo:
    def change_chk_epitope(dd_epitope):
        print(dd_epitope)
        return gr.CheckboxGroup(choices=list(dd_epitope), label='Edit epitope')

    def update_chk_epitope(sel_epitope, chks):
        sel_epitope = list(sel_epitope)
        print(sel_epitope)
        print(chks)
        if chks:
            for i in chks:
                sel_epitope[i] = '-'
        return gr.CheckboxGroup(choices=list(sel_epitope), type='index', value=chks, label='Edit epitope')

    dd_epitope = gr.Dropdown(choices=epitopes, value=sel_epitope, label='Epitope')
    chk_epitope = gr.CheckboxGroup(choices=list(sel_epitope), type='index', label='Edit epitope')
    dd_epitope.change(change_chk_epitope, inputs=dd_epitope, outputs=chk_epitope)
    chk_epitope.change(update_chk_epitope, inputs=[dd_epitope, chk_epitope], outputs=chk_epitope)

if __name__ == "__main__":
    demo.launch(debug=True, show_api=False)
