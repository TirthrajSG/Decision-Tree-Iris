import streamlit as st
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

st.write("# Predict the *IRIS* flower data using <u> Decision Tree Classifier </u>",unsafe_allow_html=True)
st.write("#")

def highlight_species(row):
    color = ''
    if row['Target'] == 'Setosa':
        color = 'background-color: #0D1B2A;'  # light blue
    elif row['Target'] == 'Versicolor':
        color = 'background-color: #1B263B ;'  # yellow
    elif row['Target'] == 'Virginica':
        color = 'background-color: #415A77 ;'  # light red
    return [color] * len(row)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
st.pyplot(plt)

label_to_name = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

df = pd.DataFrame(X, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
df["Target"] = [label_to_name[label] for label in y]
df = df.sample(frac=1, random_state=123).reset_index(drop=True)
df = df.style.apply(highlight_species, axis=1)
st.write(df, use_container_width=True)

st.sidebar.write("## Select feature of the flower to be predicted:")
sl = st.sidebar.slider("Sepal Length", max_value=5.1, min_value=0.2, value=3.0, step=0.1)
sw = st.sidebar.slider("Sepal Width", max_value=4.9, min_value=0.2, value=2.0, step=0.1)
pl = st.sidebar.slider("Petal Length", max_value=4.7, min_value=0.2, value=4.0, step=0.1)
pw = st.sidebar.slider("Petal Width", max_value=4.6, min_value=0.2, value=1.0, step=0.1)


x_ = np.array([[sl, sw, pl, pw]])
y_ = clf.predict(x_)
st.write(f'''
<h3 style='text-align: center;'> Predicted flower is <u> {[label_to_name[label] for label in y_][0]} </u>
</h3>
''', unsafe_allow_html=True)




