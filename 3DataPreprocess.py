# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:16:22 2021

@author: Alexander Humberto Nina Pacajes
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

datos_primer=pd.read_csv('jobchangepre.csv')

#1.- Remover ID
mat_feat=datos_primer.iloc[:,:-1].values
mat_class=datos_primer.iloc[:,-1].values
#print(mat_feat)
#print(mat_class)

#2.- Imputacion de datos numéricos 
imp=SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(mat_feat[:,0:2])
mat_feat[:,0:2]=imp.transform(mat_feat[:,0:2])
#print(mat_feat)
#3.- Imputacion de datos nominales en rango
imp=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(mat_feat[:,2:])
mat_feat[:,2:]=imp.transform(mat_feat[:,2:])
#print(mat_feat)
#4.- dummy code bins y encoding categorical data
#ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[2])], remainder='passthrough')
#mat_feat=np.array(ct.fit_transform(mat_feat))
#print(mat_feat)
#5.- Label Encoder
le=LabelEncoder()
mat_feat[:,2]=le.fit_transform(mat_feat[:,2])
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)
#print(mat_feat[:,2])

mat_feat[:,3]=le.fit_transform(mat_feat[:,3])
#print(mat_feat[:,3])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,4]=le.fit_transform(mat_feat[:,4])
#print(mat_feat[:,4])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,5]=le.fit_transform(mat_feat[:,5])
#print(mat_feat[:,5])
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,6]=le.fit_transform(mat_feat[:,6])
#print(mat_feat[:,6])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,7]=le.fit_transform(mat_feat[:,7])
#print(mat_feat[:,7])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,8]=le.fit_transform(mat_feat[:,8])
#print(mat_feat[:,8])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,9]=le.fit_transform(mat_feat[:,9])
#print(mat_feat[:,9])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,10]=le.fit_transform(mat_feat[:,10])
#print(mat_feat[:,10])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

mat_feat[:,11]=le.fit_transform(mat_feat[:,11])
#print(mat_feat[:,11])    
#le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

#print(mat_feat)

#6.- Separar para entrenamiento y prueba
f_entre, f_test, tar_entre, tar_test=train_test_split(mat_feat, mat_class, test_size=0.2, random_state=1)

#7.- Escalado
# =============================================================================
# escala=StandardScaler()
# f_entre[:,:]=escala.fit_transform(f_entre[:,:])
# f_test[:,:]=escala.fit_transform(f_test[:,:])
# print(f_entre)
# print(f_test)
# =============================================================================

#8.- Normalizacion
normal=Normalizer()
f_entre[:,:]=normal.fit_transform(f_entre[:,:])
f_test[:,:]=normal.fit_transform(f_test[:,:])
print(f_entre)
print(f_test)

#CSV
#df = pd.DataFrame(f_entre)
#df.to_csv(r'C:\Users\aaale\Desktop\procesado.csv')

