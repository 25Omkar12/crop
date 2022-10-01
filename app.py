
from flask import Flask, render_template,request,Markup
import pickle
import numpy as np
import os
import pandas as pd
from utils.fert_pred import fertilizer_prediction
from function import yeild_class
from werkzeug.utils import secure_filename
from keras.models import load_model
import keras
import tensorflow as tf
from keras import backend
from keras.utils import data_utils
from keras.utils import image_utils
from keras.utils import io_utils


classifier = load_model('Trained_model.h5')


with open('model_crop.pkl','rb') as f:
    mod = pickle.load(f)


app=Flask(__name__)



@app.route('/')
def trail():
    return render_template('index.html')

@app.route("/crop_pred.html")
def pred_crop():
    return render_template('crop_pred.html')

@app.route('/about.html')
def info():
    return render_template('about.html')

@app.route('/predication.html')
def pred():
    return render_template('predication.html')

@app.route('/contact.html')
def reach():
    return render_template('contact.html')

@app.route('/fertilizer.html')
def fert():
    return render_template('fertilizer.html')

@app.route('/yeild_predict.html')
def yei():
    return render_template('yeild_predict.html')

@app.route('/pesticides.html')
def pes():
    return render_template('pesticides.html')

@app.route('/manul_pest.html')
def man():
    return render_template('manul_pest.html')

@app.route('/upload.html')
def up():
    return render_template('upload.html')


def pred_pest(pest):
    try:
        test_image = tf.keras.utils.load_img(pest, target_size=(64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)
        print(result)
        return result
    except:
        return 'x'

app.config['IMAGE_UPOLADS']=r'C:\Users\omkar\Desktop\python_velocity\practise\crop\static\userfolder'
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['pest']  # fetch input
        filename = secure_filename(file.filename)

        file_path = os.path.join('static/userfolder', filename)
        file.save(file_path)


        pred = pred_pest(pest=file_path)
        print(pred)
        if pred == 'x':
            return render_template('index.html')
        if pred[0][0] == 1:
            pest_identified = 'alphids'
        elif pred[0][1] == 1:
            pest_identified = 'armyworm'
        elif pred[0][2] == 1:
            pest_identified = 'beetle'
        elif pred[0][3] == 1:
            pest_identified = 'bollworm'
        elif pred[0][4] == 1:
            pest_identified = 'earthworm'
        elif pred[0][5] == 1:
            pest_identified = 'grasshopper'
        elif pred[0][6] == 1:
            pest_identified = 'mites'
        elif pred[0][7]== 1:
            pest_identified = 'mosquito'
        elif pred[0][8] == 1:
            pest_identified = 'sawfly'
        elif pred[0][9] == 1:
            pest_identified = 'stem borer'
        

        return render_template(pest_identified + ".html")

@app.route('/pesticide',methods=["POST"])
def pesticide():
    pesti=str(request.form['pest'])
    return render_template(pesti +'.html')

@app.route('/fertilizer',methods=["post"])
def fertilizer():
    N=int(request.form['nitrogen'])
    P=int(request.form['phosphours'])
    K=float(request.form['Potassium'])
    crop_enter=str(request.form['crop_name'])

    df=pd.read_csv('Crop_NPK.csv')

    N_std=df[df['Crop']==crop_enter]['N'].iloc[0]
    P_std=df[df['Crop']==crop_enter]['P'].iloc[0]
    K_std=df[df['Crop']==crop_enter]['K'].iloc[0]

    N_diff=N-N_std
    P_diff=P-P_std
    K_diff=K-K_std

    if N_diff<0:
        value1='N_low'
        urea_need=(N_diff/(46))*100*1
    elif N_diff>0:
        value1='N_High'
        urea_need=0
    else :
        value1='N_No'

    if P_diff<0:
        value2='P_low'
        ssp_need=(P_diff/(16))*100*1
    elif P_diff>0:
        value2='K_High'
        ssp_need=0
    else :
        value2='K_No'

    if K_diff<0:
        value3='K_low'
        mop_need=(K_diff/(60))*100*1
    elif K_diff>0:
        value3='K_High'
        mop_need=0
    else :
        value3='K_No'

    recommand1=Markup(str(fertilizer_prediction[value1]))
    recommand2=Markup(str(fertilizer_prediction[value2]))
    recommand3=Markup(str(fertilizer_prediction[value3]))
    return render_template('fertilizer_pred.html',predict1=recommand1,predict2=recommand2,predict3=recommand3,u_need=urea_need,s_need=ssp_need,m_need=mop_need)

@app.route('/yeild',methods=["POST"])
def yeild():
    Area=float(request.form['Area'])
    State_name=str(request.form['State'])
    Season=str(request.form['Season'])
    Crop_name=str(request.form['Crop'])

    crop_yeild_obj=yeild_class(Area,State_name,Season,Crop_name)
    predict_yeild_crop=crop_yeild_obj.predict_yeild()

    return render_template('result_yeild.html',predict_yeild= predict_yeild_crop)


@app.route('/crop',methods=["post"])
def crop():
    N=int(request.form['nitrogen'])
    P=int(request.form['phosphours'])
    K=float(request.form['Potassium'])
    temp=float(request.form['temperature'])
    rain=float(request.form['rain'])
    ph=float(request.form['ph'])
    humidity=float(request.form['humidity'])
    data = np.array([N, P, K, temp, rain,ph, humidity],ndmin=2)

    result=mod.predict(data)

    print(result)

    return render_template('result.html',type=result[0])



if __name__ == "__main__":
    app.run(host  = '0.0.0.0' , port = 8080 ,debug=False)