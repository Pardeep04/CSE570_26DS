# CSE570: Machine Learning with Python

## Unit I: Python Fundamentals and Data Preparation for Machine Learning

### Part 1: Python Fundamentals and NumPy

---

## Unit I File Structure

1. [Part 1: Python Fundamentals and NumPy](CSE570_Unit_1_Part_1_Python_and_NumPy.md)
2. [Part 2: Pandas, Data Loading, and Missing Values](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md)
3. [Part 3: Encoding, Scaling, Splitting, and ML Pipeline](CSE570_Unit_1_Part_3_Preprocessing_and_Pipeline.md)

---

## Contents

- [1. Unit Overview](#1-unit-overview)
- [2. Learning Objectives](#2-learning-objectives)
- [3. Setting Up the Python Environment](#3-setting-up-the-python-environment)
- [4. Python Essentials](#4-python-essentials)
- [5. NumPy Arrays](#5-numpy-arrays)

---

## 1. Unit Overview

Data preparation is one of the most important stages of a machine learning project. A machine learning model can produce reliable results only when the input data is properly collected, inspected, cleaned, transformed, and divided into suitable training and testing sets.

This unit introduces:

- Essential Python concepts required for machine learning
- NumPy arrays and numerical operations
- Pandas Series and DataFrames
- Loading datasets from files
- Identifying and handling missing values
- Encoding categorical variables
- Scaling numerical features
- Splitting a dataset into training and testing sets

The topics in this unit follow the Unit I syllabus of **CSE570: Machine Learning with Python**.

---

## 2. Learning Objectives

After completing this unit, students should be able to:

- Write basic Python programs for data analysis.
- Use lists, dictionaries, loops, functions, and conditional statements.
- Create and manipulate NumPy arrays.
- Create and analyse Pandas DataFrames.
- Load CSV and Excel datasets.
- Inspect the structure and quality of a dataset.
- Detect and handle missing values.
- Convert categorical variables into numerical form.
- Scale numerical features using standard techniques.
- Divide data into training and testing subsets.
- Build a basic preprocessing workflow for machine learning.

---

## 3. Setting Up the Python Environment

### 3.1 Required Libraries

The main Python libraries used in this unit are:

- `numpy` for numerical computations
- `pandas` for data manipulation
- `matplotlib` for basic visualisation
- `scikit-learn` for preprocessing and train-test splitting
- `openpyxl` for reading Excel files

### 3.2 Installation

Run the following command in a terminal or command prompt:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

In Google Colab, most of these libraries are already installed. If required, use:

```python
!pip install numpy pandas matplotlib scikit-learn openpyxl
```

### 3.3 Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

---

## 4. Python Essentials

Python is widely used in machine learning because it has simple syntax and a large collection of data-science libraries.

### 4.1 Variables and Data Types

A variable stores a value in memory.

```python
student_name = "Aman"
age = 21
percentage = 82.5
is_eligible = True

print(student_name)
print(age)
print(percentage)
print(is_eligible)
```

Common Python data types include:

| Data Type | Description | Example |
|---|---|---|
| `int` | Integer value | `25` |
| `float` | Decimal value | `78.5` |
| `str` | Text value | `"Python"` |
| `bool` | True or False | `True` |
| `list` | Ordered collection | `[10, 20, 30]` |
| `tuple` | Immutable ordered collection | `(10, 20)` |
| `dict` | Key-value collection | `{"name": "Aman"}` |
| `set` | Unordered unique values | `{1, 2, 3}` |

The type of a variable can be checked using `type()`.

```python
marks = 85
print(type(marks))
```

### 4.2 Type Conversion

```python
age_text = "22"
age_number = int(age_text)

price = 199
price_decimal = float(price)

print(age_number, type(age_number))
print(price_decimal, type(price_decimal))
```

Type conversion is useful when data is loaded as text but must be used in numerical calculations.

### 4.3 Arithmetic Operators

```python
a = 15
b = 4

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Floor Division:", a // b)
print("Remainder:", a % b)
print("Power:", a ** b)
```

### 4.4 Comparison and Logical Operators

```python
marks = 75
attendance = 82

print(marks >= 40)
print(attendance >= 75)
print(marks >= 40 and attendance >= 75)
```

Comparison operators return Boolean values.

### 4.5 Conditional Statements

```python
marks = 68

if marks >= 75:
    grade = "Distinction"
elif marks >= 60:
    grade = "First Division"
elif marks >= 40:
    grade = "Pass"
else:
    grade = "Fail"

print("Grade:", grade)
```

Conditional statements are useful for applying rules to data.

### 4.6 Lists

A list stores multiple values in an ordered and changeable form.

```python
marks = [72, 85, 68, 90, 77]

print(marks)
print("First value:", marks[0])
print("Last value:", marks[-1])
print("First three values:", marks[:3])
```

Useful list operations:

```python
marks.append(88)
marks.remove(68)
marks.sort()

print(marks)
print("Number of values:", len(marks))
print("Maximum:", max(marks))
print("Minimum:", min(marks))
print("Sum:", sum(marks))
```

### 4.7 Tuples

A tuple is similar to a list, but its values cannot be changed after creation.

```python
coordinates = (28.61, 77.20)
print(coordinates)
print(coordinates[0])
```

### 4.8 Dictionaries

A dictionary stores information as key-value pairs.

```python
student = {
    "name": "Riya",
    "age": 20,
    "course": "Machine Learning",
    "marks": 88
}

print(student["name"])
print(student["marks"])
```

Adding and updating values:

```python
student["city"] = "Delhi"
student["marks"] = 91

print(student)
```

### 4.9 Loops

#### For Loop

```python
marks = [72, 85, 68, 90]

for value in marks:
    print(value)
```

#### While Loop

```python
count = 1

while count <= 5:
    print(count)
    count += 1
```

### 4.10 List Comprehension

List comprehension provides a concise way to create a list.

```python
numbers = [1, 2, 3, 4, 5]
squares = [number ** 2 for number in numbers]

print(squares)
```

Filtering values:

```python
marks = [32, 45, 67, 28, 81]
passed_marks = [mark for mark in marks if mark >= 40]

print(passed_marks)
```

### 4.11 Functions

A function is a reusable block of code.

```python
def calculate_average(values):
    return sum(values) / len(values)

marks = [75, 80, 90, 85]
average = calculate_average(marks)

print("Average:", average)
```

A function with a default argument:

```python
def classify_score(score, pass_marks=40):
    if score >= pass_marks:
        return "Pass"
    return "Fail"

print(classify_score(72))
print(classify_score(35))
```

### 4.12 Exception Handling

Exception handling prevents a program from stopping unexpectedly.

```python
try:
    value = int(input("Enter an integer: "))
    print("You entered:", value)
except ValueError:
    print("Invalid input. Please enter an integer.")
```

---

## 5. NumPy Arrays

NumPy stands for **Numerical Python**. It provides fast and memory-efficient arrays for numerical computations.

### 5.1 Creating a NumPy Array

```python
import numpy as np

marks = np.array([72, 85, 68, 90, 77])
print(marks)
print(type(marks))
```

### 5.2 List versus NumPy Array

Python lists can store mixed data types, while NumPy arrays usually contain values of the same type. NumPy also supports vectorised mathematical operations.

```python
python_list = [1, 2, 3]
numpy_array = np.array([1, 2, 3])

print(python_list * 2)
print(numpy_array * 2)
```

Output conceptually:

```text
[1, 2, 3, 1, 2, 3]
[2 4 6]
```

### 5.3 One-Dimensional and Two-Dimensional Arrays

```python
one_dimensional = np.array([10, 20, 30])

two_dimensional = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(one_dimensional)
print(two_dimensional)
```

### 5.4 Array Properties

```python
array = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print("Shape:", array.shape)
print("Dimensions:", array.ndim)
print("Number of elements:", array.size)
print("Data type:", array.dtype)
```

### 5.5 Special Arrays

```python
zeros = np.zeros((2, 3))
ones = np.ones((2, 3))
identity = np.eye(3)
sequence = np.arange(0, 10, 2)
evenly_spaced = np.linspace(0, 1, 5)

print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Identity matrix:\n", identity)
print("Sequence:", sequence)
print("Evenly spaced values:", evenly_spaced)
```

### 5.6 Array Indexing and Slicing

```python
values = np.array([10, 20, 30, 40, 50])

print(values[0])
print(values[-1])
print(values[1:4])
```

For a two-dimensional array:

```python
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print(matrix[0, 0])
print(matrix[1, 2])
print(matrix[:, 1])
print(matrix[0:2, 1:3])
```

### 5.7 Vectorised Operations

```python
values = np.array([10, 20, 30, 40])

print(values + 5)
print(values * 2)
print(values / 10)
print(values ** 2)
```

### 5.8 Statistical Operations

```python
marks = np.array([72, 85, 68, 90, 77])

print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard deviation:", np.std(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Sum:", np.sum(marks))
```

### 5.9 Reshaping an Array

```python
values = np.arange(1, 13)
reshaped = values.reshape(3, 4)

print(reshaped)
```

The total number of elements must remain the same during reshaping.

### 5.10 Boolean Filtering

```python
marks = np.array([32, 45, 67, 28, 81, 54])
passed = marks[marks >= 40]

print(passed)
```

### 5.11 Handling Missing Numerical Values in NumPy

NumPy represents a missing numerical value using `np.nan`.

```python
values = np.array([10, 20, np.nan, 40, 50])

print(np.isnan(values))
print("Mean ignoring missing value:", np.nanmean(values))
```

---
