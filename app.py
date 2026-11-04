from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

ROOT=Path(__file__).parent; DATA=ROOT/'data'; MODELS=ROOT/'models'
st.set_page_config(page_title='Orchard Intelligence · Shopian', page_icon='🍎', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#17241d;--muted:#728077;--green:#245b43;--lime:#b8e36a;--paper:#f5f7f2;--line:#e4e9e1}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)} .stApp{background:radial-gradient(ellipse at 85% 0%,#e9f3df 0%,transparent 34%),#f7f8f5}
[data-testid="stSidebar"]{background:linear-gradient(165deg,#152d22,#203e2e);border-right:1px solid #2e4938}
[data-testid="stSidebar"] *{color:#f1f5ed!important} [data-testid="stSidebar"] [data-testid="stRadio"] label{padding:9px 12px;border-radius:10px}
.block-container{padding-top:2.1rem;max-width:1500px} h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.035em}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:30px 34px;background:linear-gradient(120deg,#163b2a 0%,#2b6045 57%,#497b4d 100%);color:white;box-shadow:0 20px 55px #20472b22;margin-bottom:22px}
.hero:after{content:'';position:absolute;width:330px;height:330px;right:-85px;top:-145px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#c6ed7d88,#b5dc5d11 65%,transparent 70%);border:1px solid #ffffff18;box-shadow:0 0 0 35px #ffffff08,0 0 0 75px #ffffff05}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:#c6e7b3}.hero h1{font-size:34px;line-height:1.12;color:#fff;margin:8px 0 10px}.hero p{color:#d9e7dc;margin:0;max-width:730px;font-size:14px}
.glass{background:#ffffffc9;border:1px solid #e5ebe2;border-radius:18px;padding:19px 21px;box-shadow:0 7px 24px #1c35200a}
.metric-label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#718075;font-weight:700}.metric-value{font-family:Manrope,sans-serif;font-size:28px;font-weight:800;letter-spacing:-.04em;color:#1d3929;margin-top:5px}.metric-note{font-size:12px;color:#879188;margin-top:3px}
.section-head{font-family:Manrope,sans-serif;font-weight:750;font-size:20px;letter-spacing:-.03em;margin:12px 0 4px}.subtle{color:#7b887e;font-size:13px}.pill{display:inline-block;border:1px solid #dfe8dc;background:#eff5e9;border-radius:999px;padding:5px 10px;font-size:11px;color:#3d6546;font-weight:700}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e5ebe2;border-radius:16px;padding:16px 18px;box-shadow:0 6px 22px #233e2508}
.stButton>button{border-radius:11px;border:0;background:linear-gradient(115deg,#245b43,#3c7851);color:white;font-weight:700;padding:.62rem 1.1rem}.stDownloadButton>button{border-radius:10px}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid #e5ebe2;border-radius:17px;padding:9px}
hr{border-color:#e6ebe3} footer{visibility:hidden}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def csvfile(path): return pd.read_csv(path)
@st.cache_resource(show_spinner=False)
def load_m1(path): return joblib.load(path)

def hero(title,subtitle,tag='SHOPIAN · KASHMIR'):
 st.markdown(f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)
def metric(label,value,note=''):
 st.markdown(f'<div class="glass"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
def section(title,note=''):
 st.markdown(f'<div class="section-head">{title}</div><div class="subtle">{note}</div>',unsafe_allow_html=True)
def chart_theme(fig):
 fig.update_layout(template='plotly_white',font_family='DM Sans',font_color='#26382c',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=12,r=12,t=48,b=12),legend_title_text='',title_font=dict(family='Manrope',size=17),xaxis=dict(showgrid=False),yaxis=dict(gridcolor='#edf0eb'))
 return fig

with st.sidebar:
 st.markdown('<div style="font-size:10px;letter-spacing:.18em;color:#b9d9b5;font-weight:700">ORCHARD INTELLIGENCE</div><div style="font-family:Manrope;font-size:22px;font-weight:800;margin:5px 0 2px">Shopian <span style="color:#c6e77b">↗</span></div><div style="font-size:12px;color:#c2d2c4">AI for apple orchards</div>',unsafe_allow_html=True)
 st.markdown('---')
 page=st.radio('WORKSPACE',['Overview','Yield forecast','Scab weather risk','Leaf diagnosis','Data & methodology'],label_visibility='visible')
 st.markdown('---')
 st.markdown('<div style="font-size:11px;color:#c1d1c3">LOCAL PROJECT BUILD<br>Models run on this machine</div>',unsafe_allow_html=True)

# Shared local files
yp=DATA/'yield.csv'; npth=DATA/'ndvi.csv'; tp=DATA/'weather.csv'; cvp=DATA/'model1_cv_predictions.csv'; resp=DATA/'model1_cv_results.csv'
yield_df=csvfile(str(yp)) if yp.exists() else pd.DataFrame()

if page=='Overview':
 hero('Orchard intelligence,\nrooted in data.','A decision-support workspace for apple yield trends, weather-driven scab suitability, and leaf image screening.')
 if not yield_df.empty:
  latest=yield_df.sort_values('year').iloc[-1]
  c1,c2,c3,c4=st.columns(4)
  with c1: metric('Historical records',f'{len(yield_df)}','Annual yield series')
  with c2: metric('Latest year',str(int(latest.year)),'In supplied dataset')
  with c3: metric('Production',f'{latest.production_mt:,.0f} MT','Latest historical record')
  with c4: metric('Yield',f'{latest.yield_t_ha:.2f} t/ha','Latest historical record')
 st.markdown('')
 left,right=st.columns([1.35,1])
 with left:
  section('Harvest signal','Historical yield per hectare · source dataset')
  if not yield_df.empty:
   yy=yield_df.sort_values('year')
   fig=px.area(yy,x='year',y='yield_t_ha',markers=True)
   fig.update_traces(line_color='#356d4a',fillcolor='rgba(132,181,93,.17)',marker=dict(size=6,color='#356d4a'))
   st.plotly_chart(chart_theme(fig),use_container_width=True)
 with right:
  section('Three model pathways','Separate models, distinct evidence')
  st.markdown('<div class="glass"><span class="pill">MODEL 01</span><h3>Yield forecasting</h3><p class="subtle">Annual predictors · saved regression pipeline · LOYO evaluation</p><hr><span class="pill">MODEL 02</span><h3>Weather risk</h3><p class="subtle">Temperature, rainfall and humidity heuristics</p><hr><span class="pill">MODEL 03</span><h3>Leaf diagnosis</h3><p class="subtle">CNN image classification · requires trained artifact</p></div>',unsafe_allow_html=True)
 st.caption('Prototype note: weather risk is an environmental suitability estimate, not confirmed infection. Model outputs depend on supplied data and trained artifacts.')

elif page=='Yield forecast':
 hero('Yield forecast','Explore annual harvest history and estimate yield with the packaged regression model.','MODEL 01 · ANNUAL PRODUCTION')
 artifact=st.sidebar.file_uploader('Load alternate Model 1 artifact (.joblib/.pkl)',type=['joblib','pkl'],key='m1upload')
 try:
  saved=joblib.load(artifact) if artifact else load_m1(str(MODELS/'model1_yield.joblib'))
  model=saved.get('model'); features=saved.get('features',[])
  if model is None or not features: raise ValueError('Artifact should contain model and features.')
  st.success('Regression artifact loaded · '+str(saved.get('best','saved estimator')))
  if not yield_df.empty:
   fig=px.line(yield_df.sort_values('year'),x='year',y='yield_t_ha',markers=True)
   st.plotly_chart(chart_theme(fig),use_container_width=True)
  train=csvfile(str(DATA/'model1_training_table.csv')) if (DATA/'model1_training_table.csv').exists() else pd.DataFrame()
  section('Configure prediction','Keep input values within the training data range where possible.')
  cols=st.columns(2); vals={}
  for i,f in enumerate(features):
   s=pd.to_numeric(train[f],errors='coerce').dropna() if f in train else pd.Series(dtype=float)
   default=float(s.median()) if len(s) else 0.; lo=float(s.min()) if len(s) else default-100; hi=float(s.max()) if len(s) else default+100
   if lo==hi: lo-=1;hi+=1
   with cols[i%2]: vals[f]=st.number_input(f.replace('_',' ').title(),min_value=lo,max_value=hi,value=min(max(default,lo),hi),format='%.4f')
  area=st.number_input('Orchard area (hectares)',min_value=0.0,value=1.0,step=.25)
  if st.button('Generate yield estimate',type='primary'):
   pred=float(model.predict(pd.DataFrame([[vals[f] for f in features]],columns=features))[0])
   a,b=st.columns(2);a.metric('Estimated yield',f'{pred:.2f} t/ha');b.metric('Estimated production',f'{pred*area:,.2f} tonnes')
   st.caption('Model estimate, not an official forecast. Interpret with feature coverage and validation uncertainty.')
 except Exception as e: st.error(f'Model 1 unavailable: {e}')
 if cvp.exists():
  cv=csvfile(str(cvp));section('Leave-one-year-out evaluation','Actual and model predictions by held-out year')
  if {'year','actual'}.issubset(cv.columns):
   cols=['actual']+[c for c in ['Lasso','Ridge','Linear','RandomForest','Baseline_mean'] if c in cv.columns]
   fig=px.line(cv,x='year',y=cols,markers=True);st.plotly_chart(chart_theme(fig),use_container_width=True)
  st.dataframe(cv,use_container_width=True,hide_index=True)
 if resp.exists(): st.dataframe(csvfile(str(resp)),use_container_width=True,hide_index=True)

elif page=='Scab weather risk':
 hero('Weather suitability','A transparent rule-based estimate of weather conditions potentially conducive to apple scab.','MODEL 02 · WEATHER SIGNALS')
 st.warning('Not a confirmed disease diagnosis. Leaf wetness is not directly measured, and the project has no field outbreak labels for standard ML validation.')
 upload=st.file_uploader('Choose weather history CSV',type='csv',key='weather_upload')
 try:
  w=pd.read_csv(upload) if upload else csvfile(str(tp))
  required=['date','temperature_mean_c','precipitation_sum_mm','relative_humidity_mean_pct'];missing=[x for x in required if x not in w]
  if missing: st.error('Missing required columns: '+', '.join(missing))
  else:
   w['date']=pd.to_datetime(w.date,errors='coerce');w=w.dropna(subset=['date']).sort_values('date').copy()
   for col in required[1:]: w[col]=pd.to_numeric(w[col],errors='coerce')
   w['precipitation_sum_mm']=w.precipitation_sum_mm.fillna(0).clip(lower=0);w['relative_humidity_mean_pct']=w.relative_humidity_mean_pct.clip(0,100)
   w['rh3']=w.relative_humidity_mean_pct.rolling(3,min_periods=1).mean();w['rain7']=w.precipitation_sum_mm.rolling(7,min_periods=1).sum()
   rain=(w.precipitation_sum_mm>=.2).astype(int);w['consec_rain']=rain.groupby((rain!=rain.shift()).cumsum()).cumsum()
   risks=[];why=[]
   for _,r in w.iterrows():
    if not 5<=r.temperature_mean_c<=25: risk='LOW';reason='Mean temperature outside the heuristic 5–25°C activity range.'
    elif r.rh3>=80 and (r.precipitation_sum_mm>=.2 or r.consec_rain>=2):risk='HIGH';reason='Elevated rolling humidity with rain or persistent wet signal.'
    elif r.rh3>=80 or r.precipitation_sum_mm>0 or r.rain7>1:risk='MEDIUM';reason='Suitable temperature with a recent moisture indicator.'
    else:risk='LOW';reason='Suitable temperature but limited recent moisture signal.'
    risks.append(risk);why.append(reason)
   w['estimated_scab_risk']=risks;w['risk_reason']=why;w['year']=w.date.dt.year
   years=sorted(w.year.unique(),reverse=True);yr=st.selectbox('Analysis year',years);view=w[w.year==yr]
   counts=view.estimated_scab_risk.value_counts().reindex(['LOW','MEDIUM','HIGH'],fill_value=0)
   a,b,c=st.columns(3);a.metric('LOW days',int(counts.LOW));b.metric('MEDIUM days',int(counts.MEDIUM));c.metric('HIGH days',int(counts.HIGH))
   f=px.scatter(view,x='date',y='temperature_mean_c',color='estimated_scab_risk',size='precipitation_sum_mm',color_discrete_map={'LOW':'#5b946b','MEDIUM':'#d7a94a','HIGH':'#c55c51'},title=f'Daily weather risk · {yr}')
   st.plotly_chart(chart_theme(f),use_container_width=True)
   section('Daily detail','Risk explanation is generated from the rule conditions')
   st.dataframe(view[['date','temperature_mean_c','precipitation_sum_mm','relative_humidity_mean_pct','rh3','rain7','estimated_scab_risk','risk_reason']],use_container_width=True,hide_index=True)
   st.download_button('Export selected year CSV',view.to_csv(index=False).encode(),file_name=f'shopian_scab_risk_{yr}.csv',mime='text/csv')
 except Exception as e: st.error(f'Weather data could not be processed: {e}')

elif page=='Leaf diagnosis':
 hero('Leaf image screening','Upload a leaf image and the trained CNN to view its predicted class scores.','MODEL 03 · COMPUTER VISION')
 st.info('Prediction is an image-model output, not a laboratory diagnosis. Confirm suspected disease with an agricultural expert.')
 model_file=st.file_uploader('Upload trained Keras model (.keras or .h5)',type=['keras','h5'],key='leaf_model')
 image_file=st.file_uploader('Upload apple leaf image',type=['png','jpg','jpeg','webp'],key='leaf_image')
 labels=['APPLE ROT LEAVES','HEALTHY LEAVES','LEAF BLOTCH','SCAB LEAVES']
 if image_file:
  im=Image.open(image_file).convert('RGB');st.image(im,caption='Uploaded leaf',width=350)
 if model_file and image_file and st.button('Analyze leaf image',type='primary'):
  try:
   import tensorflow as tf
   import tempfile
   with tempfile.NamedTemporaryFile(suffix='.'+model_file.name.split('.')[-1]) as tmp:
    tmp.write(model_file.getvalue());tmp.flush();model=tf.keras.models.load_model(tmp.name,compile=False)
   im=Image.open(image_file).convert('RGB').resize((128,128));arr=np.asarray(im,dtype=np.float32)[None,...]
   first=model.layers[0].__class__.__name__ if model.layers else ''
   if first not in ('Rescaling','Sequential') and arr.max()>1:arr/=255.
   probs=np.asarray(model.predict(arr,verbose=0)[0]);names=labels if len(probs)==4 else [f'Class {i}' for i in range(len(probs))]
   idx=int(np.argmax(probs));st.metric('Predicted class',names[idx],f'{probs[idx]*100:.1f}% output score')
   fig=px.bar(x=names,y=probs*100,labels={'x':'Class','y':'Output score (%)'},title='Model output scores');fig.update_traces(marker_color='#4d8056');st.plotly_chart(chart_theme(fig),use_container_width=True)
  except Exception as e:st.error(f'Could not load or run CNN: {e}')
 elif not model_file: st.caption('Model artifact is not bundled in this project yet. Upload the final trained .keras/.h5 file to enable classification.')

else:
 hero('Data & methodology','Transparent data provenance and model boundaries.','PROJECT NOTES')
 section('Study area','Shopian district · Jammu & Kashmir')
 st.markdown('''**Model 1 — Yield forecasting:** Uses the saved regression pipeline and feature order in its artifact. Historical LOYO results are displayed when present.\n\n**Model 2 — Scab weather risk:** Rule-based estimation from mean temperature, precipitation, and relative humidity. Leaf wetness and observed disease outcomes are not directly available.\n\n**Model 3 — Leaf classification:** CNN image classification using the provided class mapping. Its trained binary must be supplied if not included.\n\n**Data scope:** Annual yield/production, annual NDVI summaries, and daily historical weather are included in this package. Predictions are estimates and should be interpreted within the data coverage.''')
 with st.expander('View included files'):
  for folder in [DATA,MODELS]:
   st.write(f'**{folder.name}/**')
   for f in sorted(folder.iterdir()): st.write('• '+f.name)
 st.caption('Shopian Apple Crop Intelligence · Local research prototype')
