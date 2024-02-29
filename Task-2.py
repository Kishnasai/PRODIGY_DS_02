#!/usr/bin/env python
# coding: utf-8

# In[563]:


#Importing All Required Libaries
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')


# In[565]:


img = plt.imread(r"C:\Users\dundi\OneDrive\Pictures\titanic-ship.jpg")
img = img[:, :, :3]
plt.figure(figsize=(20, 8))
grayscale_img = img[:, :, :3].mean(axis=2)
plt.figure(figsize=(12, 8))
plt.imshow(grayscale_img, cmap='gray')
plt.axis('off')
plt.title("Titanic Ship")
plt.show()


# In[566]:


df = pd.read_csv(r"C:\Users\dundi\OneDrive\Desktop\DS\Titanic_data.csv")
df.head(100)


# In[641]:


colors = ['lightgreen', '#FF69B4'] 
plt.bar(['Male', 'Female'], [male_count, female_count], color=colors)
plt.xlabel('Gender --------->')
plt.ylabel('Count --------->')
plt.title('Number of Males & Females in the Ship')
plt.show


# In[572]:


crew_count = [30, 10]
passenger_count = [40, 20]
crew_count = np.array(crew_count)
passenger_count = np.array(passenger_count)
labels = ['Male', 'Female']
fig, ax = plt.subplots()
width = 0.35
indices = np.arange(len(labels))
bar1 = ax.bar(indices - width/2, crew_count, width, label='Crew', color='#8B4513')
bar2 = ax.bar(indices + width/2, passenger_count, width, label='Passenger', color='#4169E1')
ax.set_xticks(indices)
ax.set_xticklabels(labels)
ax.set_xlabel('Gender --------->')
ax.set_ylabel('Count --------->')
ax.set_title('Male and Female Count in Crew and Passenger')
ax.legend()
plt.show()


# In[573]:


df.columns = df.columns.str.strip()
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Gender Distribution')
plt.show()


# In[584]:


plt.figure(figsize=(10, 6))
sns.violinplot(x='Class', y='Age', data=df, palette='Set1')
plt.title('Age Distribution by Class')
plt.xlabel('Class --------->')
plt.ylabel('Age --------->')
plt.show()


# In[586]:


plt.figure(figsize=(8, 6))
plt.scatter(df["Age"], df["Survival"], alpha=0.5)
plt.title("Survival by Age")
plt.xlabel("Age --------->")
plt.ylabel("Survival --------->")
plt.show()


# In[587]:


survival_gender = pd.crosstab(df['Survival'], df['Gender'])
survival_gender.plot(kind='bar', stacked=True, color=['lightblue', 'lightcoral'], figsize=(8, 6))
plt.title('Survival by Gender')
plt.xlabel('Survival --------->')
plt.ylabel('Count --------->')
plt.show()


# In[596]:


plt.figure(figsize=(10, 6))
sns.kdeplot(x='Age', hue='Survival', data=df, fill=True, palette='Set2')
plt.title('Age Distribution by Survival')
plt.xlabel('Age --------->')
plt.show()


# In[593]:


# Create a swarm plot of Age distribution by Class
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Class', y='Age', data=df, palette='Set1')
plt.title('Age Distribution by Class')
plt.xlabel('Class --------->')
plt.ylabel('Age --------->')
plt.show()


# In[613]:


survival_counts = df['Survival'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(survival_counts, labels=survival_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ADD8E6', '#8B1A00'])
plt.title('Survival Distribution')
plt.show()


# In[614]:


plt.figure(figsize=(12, 8))
sns.swarmplot(x='Gender', y='Age', hue='Class', data=df, palette='viridis')
plt.title('Age Distribution by Gender and Class')
plt.xlabel('Gender --------->')
plt.ylabel('Age --------->')
plt.show()


# In[616]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='Gender', y='Age', hue='Survival', data=df, palette='Set1')
plt.title('Age Distribution by Gender and Survival')
plt.xlabel('Gender --------->')
plt.ylabel('Age --------->')
plt.show()


# In[623]:


plt.figure(figsize=(12, 8))
sns.lineplot(x='Class', y='Age', data=df, marker='o', ci=None, color='#E91E63')
plt.title('Age Trends over Classes')
plt.xlabel('Class --------->')
plt.ylabel('Average Age --------->')
plt.show()


# In[632]:


plt.figure(figsize=(12, 8))
sns.pointplot(x='Class', y='Age', hue='Gender', data=df, palette='Dark2')
plt.title('Age Distribution by Class and Gender')
plt.xlabel('Class --------->')
plt.ylabel('Average Age --------->')
plt.show()


# In[635]:


plt.figure(figsize=(12, 8))
sns.stripplot(x='Role', y='Age', data=df, palette='Set1', jitter=True)
plt.title('Age Distribution by Role')
plt.xlabel('Role --------->')
plt.ylabel('Age --------->')
plt.show()


# In[643]:


survival_gender_counts = df.groupby(['Survival', 'Role']).size().unstack()
survival_gender_counts.plot(kind='pie', subplots=True, figsize=(15, 8), autopct='%1.1f%%', startangle=90,
                            colors=['lightgreen', 'coral'], wedgeprops=dict(width=0.3), legend=False)
plt.title('Survival Distribution by Gender')
plt.show()


# In[645]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette='Set3')
plt.title('Class Distribution')
plt.xlabel('Class --------->')
plt.ylabel('Count --------->')
plt.show()


# In[648]:


plt.figure(figsize=(10, 6))
sns.swarmplot(x='Survival', y='Age', data=df, palette='Set1')
plt.title('Age Distribution by Survival')
plt.xlabel('Survival --------->')
plt.ylabel('Age --------->')
plt.show()


# In[654]:


contingency_table = pd.crosstab(df["Class"], df["Survival"])
normalized_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)
plt.figure(figsize=(10, 6))
sns.heatmap(normalized_table, annot=True, fmt=".2f")
plt.title("Survival Rate by Class (Heatmap)")
plt.show()


# In[656]:


plt.figure(figsize=(10, 6))
sns.lineplot(x="Class", y="Age", hue="Survival", data=df)
plt.title("Age Distribution by Class and Survival")
plt.show()


# In[366]:


img = plt.imread(r"C:\Users\dundi\OneDrive\Pictures\titanic-sink.jpg")
img = img[:, :, :3]
plt.figure(figsize=(30, 6))
grayscale_img = img[:, :, :3].mean(axis=2)
plt.figure(figsize=(12, 8))
plt.imshow(grayscale_img, cmap='gray')
plt.axis('off')
plt.title("Titanic Sink")
plt.show()

