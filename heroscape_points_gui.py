#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import pathlib, joblib, pandas as pd, numpy as np, re, sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import sys
import os



TYPE_MAP = {0:"Unique Hero",1:"Common Hero",2:"Unique Squad",3:"Common Squad",4:"Event Hero"}
TYPE_LABELS = list(TYPE_MAP.values())

# ------------ load model ----------------
if hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    MODEL_PATH = os.path.join(sys._MEIPASS, "heroscape_point_model.pkl")
    BG_PATH    = os.path.join(sys._MEIPASS, "heroscape_bg.png")
    CSV_PATH   = os.path.join(sys._MEIPASS, "heroscape_characters.csv")
else:
    # Running in normal Python environment
    MODEL_PATH = pathlib.Path("heroscape_point_model.pkl")
    BG_PATH    = pathlib.Path("heroscape_bg.png")
    CSV_PATH   = pathlib.Path("heroscape_characters.csv")

# if not MODEL_PATH.exists():
#     messagebox.showerror("Model missing", f"{MODEL_PATH} not found"); sys.exit(1)
model = joblib.load(MODEL_PATH)

# numeric list from pipeline, but we’ll add logHeight ourselves
NUMERIC_COLS = ["Life","MVE","RGE","ATK","DEF","logHeight","NumUnits"]

# ------------ dropdown choices ----------
if CSV_PATH:
    df_src = pd.read_csv(CSV_PATH)
    if df_src["Type"].dtype != object:
        df_src["Type"] = df_src["Type"].map(TYPE_MAP)
else:
    df_src = pd.DataFrame()

cat_opts = {c: sorted(df_src[c].dropna().unique()) for c in ("Army","Race","Class","Size","Type")}
cat_opts["Type"] = TYPE_LABELS
cat_default = {k:(v[0] if v else "") for k,v in cat_opts.items()}

# ------------ helper --------------------
def build_df(v):
    row = {
        "Life":v["Life"], "MVE":v["MVE"], "RGE":v["RGE"],
        "ATK":v["ATK"],   "DEF":v["DEF"],
        "logHeight": np.log1p(v["Height"]),
        "NumUnits": v["NumUnits"],
        "Abilities": v["Abilities"],
        "Army":v["Army"], "Race":v["Race"], "Class":v["Class"],
        "Size":v["Size"], "Type":v["Type"]
    }
    order = NUMERIC_COLS + ["Army","Race","Class","Size","Type","Abilities"]
    return pd.DataFrame([row], columns=order)

# ------------ GUI -----------------------
root = tk.Tk(); root.title("HeroScape Point Predictor"); root.minsize(750,550)

if BG_PATH:
    raw = Image.open(BG_PATH); bg = ImageTk.PhotoImage(raw)
    lbl = tk.Label(root,image=bg); lbl.place(relwidth=1,relheight=1)
    def _res(ev):
        img = raw.resize((ev.width,ev.height), Image.LANCZOS)
        lbl.img = ImageTk.PhotoImage(img); lbl.config(image=lbl.img)
    root.bind("<Configure>",_res)

frm = ttk.Frame(root,padding=15); frm.place(x=20,y=20)

num_fields = ["Life","MVE","RGE","ATK","DEF","Height","NumUnits"]; num_vars={}
for r,f in enumerate(num_fields):
    ttk.Label(frm,text=f+':').grid(row=r,column=0,sticky='e')
    v=tk.IntVar(value=1 if f=="NumUnits" else 0); num_vars[f]=v
    ttk.Entry(frm,width=7,textvariable=v).grid(row=r,column=1,sticky='w')

cat_vars={}; base=len(num_fields)
for i,c in enumerate(("Army","Race","Class","Size","Type")):
    ttk.Label(frm,text=c+':').grid(row=base+i,column=0,sticky='e')
    var=tk.StringVar(value=cat_default.get(c,"")); cat_vars[c]=var
    ttk.Combobox(frm,textvariable=var,values=cat_opts[c],width=18)\
        .grid(row=base+i,column=1,sticky='w')

ttk.Label(frm,text="Abilities (full text or names; semicolons OK):")\
   .grid(row=0,column=2,sticky='nw',padx=(15,0))
abilities = scrolledtext.ScrolledText(frm,width=45,height=14,wrap=tk.WORD)
abilities.grid(row=1,column=2,rowspan=12,padx=(15,0))

result=tk.StringVar(value="Enter stats and click Predict")
ttk.Label(frm,textvariable=result,font=("Georgia",12,"bold"))\
   .grid(row=base+7,column=0,columnspan=3,pady=10)

def predict():
    try:
        v={f:num_vars[f].get() for f in num_fields}
        v.update({c:cat_vars[c].get() for c in cat_vars})
        v["Abilities"]=abilities.get("1.0",tk.END).strip()
        df=build_df(v)
        cpp=float(model.predict(df)[0])
        total=cpp*max(1,v["NumUnits"])
        result.set(f"Predicted Point Cost: {round(total)}")
    except Exception as e:
        messagebox.showerror("Error",str(e))

ttk.Button(frm,text="Predict",command=predict)\
   .grid(row=base+6,column=0,columnspan=2,pady=5)

root.mainloop()
