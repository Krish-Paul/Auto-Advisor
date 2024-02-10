# # -*- coding: utf-8 -*-
# """
# Created on Sun Feb  4 09:28:03 2024

# @author: KRISHNANGSHU
# """
# import pickle
# import numpy as np
# import streamlit as st

# mod=pickle.load(open('mod.pkl','rb'))
# st.title('Car Budget Price Prediction using ML')
# col1, col2, col3, col4, col5= st.columns(5)
# with col1:
#     gender = st.text_input("Male(1) || Female(0)")
# gender1=int(gender)
        
# with col2:
#     age = st.text_input("Age")
# age1=int(age)

# with col3:
#     salary = st.text_input("Salary")
# salary1=int(salary)

# with col4:
#     cred = st.text_input("Credit Card Debt")
# cred1=int(cred)

# with col5:
#     net = st.text_input("Net Worth")
# net1=int(net)
# val=np.array([[gender,age,salary,cred,net]])
# budget=mod.predict(val)
# st.success("The expected value ",budget)
import pickle
import numpy as np
import streamlit as st

mod = pickle.load(open('mod.pkl', 'rb'))
st.title('Auto Advisor: Know your Budget')
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    gender = st.text_input(" Male(1) || Female(0)")
    gender1 = int(gender) if gender else 0
        
with col2:
    age = st.text_input("Age")
    age1 = int(age) if age else 0

with col3:
    salary = st.text_input("Salary")
    salary1 = int(salary) if salary else 0

with col4:
    cred = st.text_input("Credit Card Debt")
    cred1 = int(cred) if cred else 0

with col5:
    net = st.text_input("Net Worth")
    net1 = int(net) if net else 0

val = np.array([[gender1, age1, salary1, cred1, net1]])
budget = mod.predict(val)

st.success(f"The expected value: {budget[0]}")
