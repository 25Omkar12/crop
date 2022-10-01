from distutils.command.config import config
import pickle
from unittest import result
import numpy as np
import json
import bz2
import pickle
import pickle as cPickle




class yeild_class():
    def __init__(self, Area,State_name,Season,Crop_name):
        self.Area=Area
        self.State_name=State_name
        self.Season=Season
        self.Crop_name=Crop_name
    def load_model(self):
        data = bz2.BZ2File('com_mod.pbz2', 'rb')
        self.model = cPickle.load(data)
            
        with open('columns_list.json','r') as file:
            self.columns_dict = json.load(file)
    
    def predict_yeild(self):
        self.load_model()
        array = np.zeros(len(self.columns_dict['columns']))
        array[0]=self.Area
        
        state_name='State_Name_'+self.State_name
        state_index=self.columns_dict['columns'].index(state_name)
        array[state_index]=1
        
        
        season_name='Season_' + self.Season
        season_index=self.columns_dict['columns'].index(season_name)
        array[season_index]=1


        Crop_name= 'Crop_' + self.Crop_name
        crop_index=self.columns_dict['columns'].index(Crop_name)
        array[crop_index]=1

        result = self.model.predict([array])
        print("okay")
        return result[0]


if __name__ == '__main__':
    Area=720.0
    State_name='AndamanandNicobarIslands'
    Season='WholeYear'
    Crop_name='Cashewnut'
    
    crop_yeild_obj=yeild_class(Area,State_name,Season,Crop_name)
    
    crop_yeild_obj.predict_yeild()