# import the necessary packages 
from __future__ import division, print_function
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import os
import glob
import re
import numpy as np
import argparse
import cv2
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import image , img_to_array, load_img
import base64
from flask import Flask, redirect, url_for, request, render_template,make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import time
import psycopg2
import json


# Define a flask app
app = Flask(__name__)


def getPhoto2(data):

  cat_records = tuple(data)
  breed_search = (cat_records,)
  photo_data = []
  photo_data2 = []
  try:
    connection = psycopg2.connect(user="postgres",
                                  password="unknow544",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="postgres")
    cursor = connection.cursor()
    

    query2 = """ SELECT photo
                  from "PHOTO"
                  where id_photo in 
                    (select MIN(id_photo) 
                    FROM "PHOTO" 
                    where breed_name_eng in %s 
                    GROUP BY breed_name_eng)
                  order by breed_name_eng """
    cursor.execute(query2,breed_search)
    photo_records = cursor.fetchall() 

    for row in photo_records:
 
      binary_img = row[0].tobytes()
      base64_img = base64.b64encode(binary_img)
      photo_data.append(base64_img)


    data = ''.join(str(x) for x in photo_data)
    photo =  data.split("'")

    for i in range(0, len(photo)):
      if(photo[i] != 'b'):
        photo_data2.append(photo[i])

  except (Exception, psycopg2.Error) as error :
      print ("Error while fetching data from PostgreSQL", error)
  finally:
    #closing database connection.
      if(connection):
          cursor.close()
          connection.close()

  return photo_data2


def getPhoto(data):

  breed_search = (data,)
  photo_data = []
  try:
    connection = psycopg2.connect(user="postgres",
                                  password="unknow544",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="postgres")
    cursor = connection.cursor()

    query2 = """ SELECT photo FROM "PHOTO" WHERE breed_name_eng = %s """
    cursor.execute(query2, breed_search)
    photo_records = cursor.fetchall() 
 
    for row in photo_records:
 
      binary_img = row[0].tobytes()
      base64_img = base64.b64encode(binary_img)
      photo_data.append(base64_img)

  except (Exception, psycopg2.Error) as error :
      print ("Error while fetching data from PostgreSQL", error)
  finally:
    #closing database connection.
      if(connection):
          cursor.close()
          connection.close()

  data = ','.join(str(x) for x in photo_data)
 
  return data



def model_predict2(img_path):
    top_model_weights_path = 'models/weight_last_MobileNet.h5'
    class_dictionary = np.load('models/class_indices.npy').item()
    num_classes = len(class_dictionary)
    img = image.load_img(img_path, target_size=(224,224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # build Pre-trained
    model = applications.MobileNet(include_top=False, weights='imagenet', input_shape=(224,224,3))
    x = preprocess_input(x, mode='tf')

    # get the bottleneck prediction from the pre-trained model
    bottleneck_prediction = model.predict(x)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='softsign'))
    model.add(Dropout(0.58))
    model.add(Dense(num_classes, activation='softmax'))
    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    probabilities = model.predict_proba(bottleneck_prediction)
    max_pro = round(np.max(probabilities)*100, 2)
    inID = class_predicted[0]
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]
    print(label)

    #acc > 70 --->show
    if(max_pro >= 65):
      max_pro = str(max_pro)+"%"
    else:
      max_pro = ""
      label = "ไม่พบแมว"

    return label+"'"+max_pro

def model_predict(img_path):
    top_model_weights_path = 'models/weight_last_VGG16.h5'
    class_dictionary = np.load('models/class_indices.npy').item()
    num_classes = len(class_dictionary)
    img = image.load_img(img_path, target_size=(224,224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # build the Pretrained
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
    x = preprocess_input(x, mode='tf')

    # get the bottleneck prediction from the pre-trained model
    bottleneck_prediction = model.predict(x)
    
    
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='softplus'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    probabilities = model.predict_proba(bottleneck_prediction)
    max_pro = round(np.max(probabilities)*100, 2)


    inID = class_predicted[0]
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]
    print(label)

    if(max_pro >= 65):
      max_pro = str(max_pro)+"%"
    else:
      max_pro = ""
      label = "ไม่พบแมว"

    return label+"'"+max_pro



@app.route('/Classifier', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict_capture_time', methods=['GET', 'POST'])
def capture_time():
  
    f = request.form['keyword']
    img_cat = f.split(",")
    imgdata = base64.b64decode(img_cat[1])
    filename = time.strftime("%Y%m%d-%H%M%S")+".jpg"

    with open("uploads/"+filename, 'wb') as f:
        f.write(imgdata)
    f.close()
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
            basepath, 'uploads', secure_filename(filename))

    preds_max = model_predict(file_path)
    preds =  preds_max.split("'")

    if(preds[0] == "Thai"):
        photo = getPhoto("Siam") 
       
    else:
        photo = getPhoto(preds[0]) 


        data = preds_max +"'"+ photo

    return data 

@app.route('/predict_capture_acc', methods=['GET', 'POST'])
def capture_acc():
  
    f = request.form['keyword']
    img_cat = f.split(",")
    imgdata = base64.b64decode(img_cat[1])
    filename = time.strftime("%Y%m%d-%H%M%S")+".jpg"

    with open("uploads/"+filename, 'wb') as f:
        f.write(imgdata)
    f.close()
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
            basepath, 'uploads', secure_filename(filename))

    preds_max = model_predict2(file_path)
    preds =  preds_max.split("'")
 
    if(preds[0] == "Thai"):
        photo = getPhoto("Siam") 
       
    else:
        photo = getPhoto(preds[0]) 


        data = preds_max +"'"+ photo

   
    return data 

@app.route('/predict_img_time', methods=['GET', 'POST'])
def upload_img_time():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds_max = model_predict(file_path)
        preds =  preds_max.split("'")
    
        if(preds[0] == "Thai"):
          photo = getPhoto("Siam") 

        else:
          photo = getPhoto(preds[0]) 


        data = preds_max +"'"+ photo
   

        return data
  
    return None

@app.route('/predict_img_acc', methods=['GET', 'POST'])
def upload_img_acc():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds_max = model_predict2(file_path)
        preds =  preds_max.split("'")

        if(preds[0] == "Thai"):
          photo = getPhoto("Siam") 

        else:
          photo = getPhoto(preds[0]) 

        data = preds_max +"'"+ photo
  
        return data
  
    return None

@app.route('/StartAdvanceSearch', methods=['GET'])
def StartCatInformationSearch():
      data = []
      data.append("%ข%")
      data.append("%ข%")
      data.append("%ข%")

      return render_template('CatInformationSearch.html',rows=data)

def getData(data):

  try:
    connection = psycopg2.connect(user="postgres",
                                  password="unknow544",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="postgres")
    cursor = connection.cursor()
    query1 = """ SELECT * FROM "CATBREED" WHERE breed_name_eng = %s ORDER BY breed_name_eng"""
    breed_search = (data,)
    cursor.execute(query1, breed_search,)
    data_records = cursor.fetchall() 

    query2 = """ SELECT photo FROM "PHOTO" WHERE breed_name_eng = %s """
    cursor.execute(query2, breed_search)
    photo_records = cursor.fetchall() 

    for row in data_records:
      cat_data = row[0]+"'"+row[1]+"'"+row[2]+"'"+row[3]+"'"+row[4]+"'"

    photo_data = []
    for row in photo_records:
   
      binary_img = row[0].tobytes()
      base64_img = base64.b64encode(binary_img)
      photo_data.append(base64_img)

  except (Exception, psycopg2.Error) as error :
      print ("Error while fetching data from PostgreSQL", error)
  finally:
    #closing database connection.
      if(connection):
          cursor.close()
          connection.close()

  cat_records = ','.join(str(x) for x in photo_data)
  data = cat_data+cat_records
  return data

@app.route('/datasearch', methods=['GET', 'POST'])
def advancesearch():
  color = request.args.get('catcolor')
  size = request.args.get('catsize')
  fur = request.args.get('catfur')
  cat = []
  breed_name = []
  cat.append(color)
  cat.append(size)
  cat.append(fur)

  try:
    connection = psycopg2.connect(user="postgres",
                                  password="unknow544",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="postgres")
    cursor = connection.cursor()
    query1 = """ SELECT * FROM "CATBREED" WHERE appearance LIKE %s and appearance LIKE %s and appearance LIKE %s ORDER BY breed_name_eng"""
    breed_search = (color,size,fur,)
    cursor.execute(query1, breed_search)
    data_records = cursor.fetchall()
    cat.extend(data_records)

    for row in data_records:
      breed_name.append(row[0])
    photo = getPhoto2(breed_name)

    if(len(data_records) > 0):
      cat_data = cat
    else:
      cat.append("n")
      cat_data = cat


  except (Exception, psycopg2.Error) as error :
      print ("Error while fetching data from PostgreSQL", error)
  finally:
    #closing database connection.
      if(connection):
          cursor.close()
          connection.close()
  
  return render_template("CatInformationSearch.html", rows=cat_data,data = photo)



@app.route('/datasearch2', methods=['GET', 'POST'])
def advancesearch2():
  color = request.args.get('catcolor')
  size = request.args.get('catsize')
  fur = request.args.get('catfur')
  cat = []
  cat.append(color)
  cat.append(size)
  cat.append(fur)

  try:
    connection = psycopg2.connect(user="postgres",
                                  password="unknow544",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="postgres")
    cursor = connection.cursor()
    query1 = """ SELECT * FROM "CATBREED" WHERE appearance LIKE %s and appearance LIKE %s and appearance LIKE %s ORDER BY breed_name_eng"""
    breed_search = (color,size,fur,)
    cursor.execute(query1, breed_search)
    data_records = cursor.fetchall()
    cat.extend(data_records)
 
    if(len(data_records) > 0):
      cat_data = cat
    else:
      cat.append("n")
      cat_data = cat


  except (Exception, psycopg2.Error) as error :
      print ("Error while fetching data from PostgreSQL", error)
  finally:
    #closing database connection.
      if(connection):
          cursor.close()
          connection.close()
  
  return render_template("CatInformationSearch.html", rows=cat_data)

@app.route('/StartCatInformation', methods=['GET', 'POST'])
def StartCatInformation():
    return render_template('CatInformation.html')


@app.route('/dataget', methods=['GET', 'POST'])
def catinformation():
    data = request.form['keyword']
    data2 = getData(data)
  
    return data2


@app.route('/DataCat', methods=['GET', 'POST'])
def showInfoCat():

  breedeng = request.args.get('breedeng')
  breedthai = request.args.get('breedthai')
  apperance = request.args.get('apperance')
  habit = request.args.get('habit')
  takecare = request.args.get('takecare')
  

  data = getPhoto(breedeng)
  photo = data.split("'")
  data=[
    {
      'breedeng':breedeng,
      'breedthai': breedthai,
      'apperance': apperance,
      'habit':habit,
      'takecare': takecare,
      'photo1' : photo[1],
      'photo2' : photo[3],
      'photo3' : photo[5],
      'photo4' : photo[7]


    }]


  
  return render_template("ShowInfoCat.html",data=data) 

@app.route('/showData/<breed>', methods=['GET', 'POST'])
def showData(breed):
  
  data = getData(breed)
  info = data.split("'")
  data_cat=[
    {
      'breedeng':info[0],
      'breedthai': info[1],
      'apperance': info[2],
      'habit':info[3],
      'takecare': info[4],
      'photo1' : info[6],
      'photo2' : info[8],
      'photo3' : info[10],
      'photo4' : info[12]


    }]

  
  return render_template("ShowInfoCat.html" , data=data_cat) 



if __name__ == '__main__':
    #app.run(port=5002, debug=True)

    #Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
