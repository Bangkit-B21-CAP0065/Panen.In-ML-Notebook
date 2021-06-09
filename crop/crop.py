import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
dirname = os.path.dirname(__file__)

# variabel input user : subround,location,crop
# model : model_bandung,model_bogor,model_rice,model_corn

# Dataframe for forecast and crop prediction
df_bandung  = pd.read_csv(os.path.join(dirname, 'df_bandung.csv'))
df_bogor = pd.read_csv(os.path.join(dirname, 'df_bogor.csv'))

# Model for prediction
model_bandung = tf.keras.models.load_model(os.path.join(dirname, 'model_bandung.h5'))
model_bogor = tf.keras.models.load_model(os.path.join(dirname, 'model_bogor.h5'))
model_rice  = tf.keras.models.load_model(os.path.join(dirname, 'model_rice.h5'))
model_corn  = tf.keras.models.load_model(os.path.join(dirname, 'model_corn.h5'))

# Climate Forecast Function
def forecast(subround,location,model_bandung=model_bandung,model_bogor=model_bogor) :
    output  =  []
    width = 4
    features =  ['Tn','Tx','Tavg','RH_avg','RR','ss','ff_avg','ff_x','ff_y']

    df_bandung_copy = df_bandung.copy()
    df_bogor_copy = df_bogor.copy()

    if location == 'Kab.Bandung'  :
        for i in range(subround):
            input = df_bandung_copy[-width:]
            input_forecast = np.array(input)[np.newaxis]
            temp = model_bandung.predict(input_forecast)
            temp = np.squeeze(temp, axis=None)
            output.append(temp)
            df_bandung_copy = df_bandung_copy.append(pd.DataFrame([temp], columns=features))
    if location == 'Kab.Bogor'  :
        for i in range(subround):
            input = df_bogor_copy[-width:]
            input_forecast = np.array(input)[np.newaxis]
            temp = model_bandung.predict(input_forecast)
            temp = np.squeeze(temp, axis=None)
            output.append(temp)
            df_bogor_copy = df_bandung_copy.append(pd.DataFrame([temp], columns=features))
    return output

def spaceremoval(input):
        #space problem
        result=""
        count=1
        for item in input.split():
            if count==1 or count==len(input.split()):
                result+=item
            else:
                result+=" "+item
            count+=1
        return result

# Main Function
def main():
    subround = 5
    location = 'Kab.Bandung'
    crop = 'Corn'

    try:
      climate = np.array(forecast(subround,location))
      # print(climate)
      # print('')
      if crop == 'Rice' :
          output  = np.array(model_rice.predict(climate))
      elif crop  == 'Corn':
          output  = np.array(model_corn.predict(climate))
      else  :
          return "Invalid Input"

    except ValueError:
        return "Please Enter valid values"

    output = np.squeeze(output, axis=None)
    output = spaceremoval(str(output))
    print(output)

main()
