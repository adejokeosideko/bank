import streamlit as st
import pandas as pd
import pickle
with open("LR_model.pkl", "rb") as f:
    LR_model= pickle.load(f)
with open("LR_model.pkl", "rb")as f:
    scalar = pickle.load(f)