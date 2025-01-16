#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:24:29 2024

@author: dex
"""

#MATPLOTLIB FOR PLOTS

import matplotlib.pyplot as plt
import numpy as np

'''
x = np.linspace(0,10,100)

y = np.sin(x)

z = np.cos(x)

plt.plot(x,y)
plt.show()
plt.plot(x,z)
plt.show()

#ADDING X LABELS AND Y LABELS
plt.plot(x,y)
plt.xlabel("Angel")
plt.ylabel("Sine Values")
plt.title("Sine Wave")
plt.show()
'''
#PARABOLA

x = np.linspace(-10,10,20)
y = x ** 2
plt.plot(x,y, "r x")            #"r +", "g ."...
plt.show()

x = np.linspace(-5,5,50)
plt.plot(x, np.sin(x), "g .")
plt.plot(x, np.cos(x), "r +")
plt.show()

#BAR PLOT

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Languages = ["English", "French", "Yoruba", "Igbo", "Hausa"]
people = [200,300,500,400,600]
ax.bar(Languages, people)
plt.xlabel("LANGUAGES")
plt.ylabel("NUMBER OF PEOPLE")
plt.show()


#PIE CHART

fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
Languages = ["English", "French", "Yoruba", "Igbo", "Hausa"]
people = [200,300,500,400,600]
ax.pie(people, labels = Languages, autopct = "%1.1f%%")
plt.show()


#SCATTERPLOT

x = np.linspace(0,10,30)
y = np.sin(x)
z = np.cos(x)
fig2 = plt.figure()
ax = fig2.add_axes([0,0,1,1])
ax.scatter(x,y, color = "g")
plt.scatter(x,z,color = "b")

plt.show()

#3D SCATTERPLOT

fig3 = plt.figure()
ax = plt.axes(projection = "3d")
z = 20 * np.random.random(100)
x = np.sin(z)
y = np.cos(z)

ax.scatter(x,y,z,c =z, cmap = "Blues")
plt.show()










