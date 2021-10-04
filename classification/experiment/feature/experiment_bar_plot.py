import matplotlib.pyplot as plt
import numpy as np

a = np.arange(-4, 4)
b = np.arange(5)

langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]


plt.bar(langs, students, color ='maroon',
        width = 0.4)

plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()

print()