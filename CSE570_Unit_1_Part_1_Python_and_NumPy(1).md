# 🐍 CSE570: Machine Learning with Python

## Unit I — Part 1: Python Fundamentals and NumPy

> **Course focus:** Python essentials and NumPy arrays required for machine learning.

---

## 📚 Unit I Navigation

| Part | Topic | File |
|---|---|---|
| **1** | Python Fundamentals and NumPy | **You are here** |
| **2** | Pandas, Data Loading, and Missing Values | [Open Part 2](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md) |
| **3** | Encoding, Scaling, Splitting, and ML Pipeline | [Open Part 3](CSE570_Unit_1_Part_3_Preprocessing_and_Pipeline.md) |

---

## 🎯 Learning Outcomes

After completing this file, students will be able to:

- explain why Python is widely used in machine learning;
- create variables using appropriate data types;
- apply arithmetic, comparison, and logical operators;
- use conditions, loops, lists, tuples, dictionaries, and functions;
- create and manipulate NumPy arrays;
- perform indexing, slicing, reshaping, aggregation, and vectorised operations.

---

<table><tr><td bgcolor="#E8F4FD">
<b>💡 Why this topic matters</b><br>
Machine learning code is built using Python instructions. Before working with datasets and algorithms, students must be comfortable with variables, collections, conditions, loops, functions, and numerical arrays.
</td></tr></table>

## 1. Setting Up the Environment

### 1.1 Required libraries

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

### 1.2 Importing NumPy

```python
# Import NumPy and give it the standard short name "np".
# The alias allows us to write np.array() instead of numpy.array().
import numpy as np

# Display the installed NumPy version.
print("NumPy version:", np.__version__)
```

<table><tr><td bgcolor="#FFF4CE">
<b>⚠️ Student note</b><br>
Run the installation command only when a library is unavailable. Google Colab already contains most common data-science libraries.
</td></tr></table>

---

# 2. Python Essentials

## 2.1 Variables and Data Types

A **variable** is a named location used to store a value. Python automatically identifies the type of value assigned to a variable.

```python
# A string stores text.
student_name = "Aman"

# An integer stores a whole number.
age = 21

# A float stores a decimal number.
percentage = 82.5

# A Boolean stores either True or False.
is_eligible = True

# Print all stored values.
print("Name:", student_name)
print("Age:", age)
print("Percentage:", percentage)
print("Eligible:", is_eligible)
```

### Common data types

| Data type | Meaning | Example |
|---|---|---|
| `int` | Whole number | `25` |
| `float` | Decimal number | `78.5` |
| `str` | Text | `"Python"` |
| `bool` | Logical value | `True` |
| `list` | Ordered, changeable collection | `[10, 20, 30]` |
| `tuple` | Ordered, fixed collection | `(10, 20)` |
| `dict` | Key–value collection | `{"name": "Aman"}` |
| `set` | Unordered collection of unique values | `{1, 2, 3}` |

```python
marks = 85

# type() tells us which data type Python has assigned.
print(type(marks))
```

**Expected output:**

```text
<class 'int'>
```

---

## 2.2 Type Conversion

Data read from forms or files may be stored as text. Type conversion changes it into a form suitable for calculation.

```python
# This value looks numeric, but it is currently text.
age_text = "22"

# Convert the string into an integer.
age_number = int(age_text)

# Convert an integer into a floating-point value.
price = 199
price_decimal = float(price)

print(age_number, type(age_number))
print(price_decimal, type(price_decimal))
```

<table><tr><td bgcolor="#FDECEC">
<b>❌ Common mistake</b><br>
<code>"20" + "5"</code> produces <code>"205"</code> because both values are strings. Convert them using <code>int()</code> before addition.
</td></tr></table>

```python
first_value = "20"
second_value = "5"

# Convert both strings before performing numerical addition.
total = int(first_value) + int(second_value)
print(total)  # 25
```

---

## 2.3 Arithmetic Operators

```python
a = 15
b = 4

print("Addition:", a + b)          # 19
print("Subtraction:", a - b)       # 11
print("Multiplication:", a * b)    # 60
print("Division:", a / b)          # 3.75
print("Floor division:", a // b)   # 3
print("Remainder:", a % b)         # 3
print("Power:", a ** b)            # 15 raised to power 4
```

| Operator | Operation | Example |
|---|---|---|
| `+` | Addition | `5 + 2` |
| `-` | Subtraction | `5 - 2` |
| `*` | Multiplication | `5 * 2` |
| `/` | Decimal division | `5 / 2` |
| `//` | Floor division | `5 // 2` |
| `%` | Remainder | `5 % 2` |
| `**` | Exponentiation | `5 ** 2` |

---

## 2.4 Comparison and Logical Operators

Comparison operators return `True` or `False`.

```python
marks = 75
attendance = 82

# Check each academic condition separately.
passed_exam = marks >= 40
has_attendance = attendance >= 75

# 'and' is True only when both conditions are True.
is_eligible = passed_exam and has_attendance

print("Passed exam:", passed_exam)
print("Attendance complete:", has_attendance)
print("Eligible:", is_eligible)
```

| Operator | Meaning |
|---|---|
| `==` | Equal to |
| `!=` | Not equal to |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal to |
| `<=` | Less than or equal to |
| `and` | Both conditions must be true |
| `or` | At least one condition must be true |
| `not` | Reverses a Boolean value |

---

## 2.5 Conditional Statements

Conditional statements allow a program to make decisions.

```python
marks = 68

# Python checks conditions from top to bottom.
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

<table><tr><td bgcolor="#EAF7EA">
<b>✅ Key idea</b><br>
Indentation is compulsory in Python. The indented statements belong to the corresponding <code>if</code>, <code>elif</code>, or <code>else</code> block.
</td></tr></table>

---

## 2.6 Lists

A list stores multiple values in an ordered and changeable collection.

```python
marks = [72, 85, 68, 90, 77]

# Python indexing starts from zero.
print("First mark:", marks[0])

# A negative index counts from the end.
print("Last mark:", marks[-1])

# Slicing includes the start position but excludes the end position.
print("First three marks:", marks[0:3])

# Add a new value at the end.
marks.append(88)

# Replace the value at index 2.
marks[2] = 70

print("Updated list:", marks)
```

### Useful list functions

```python
marks = [72, 85, 68, 90, 77]

print("Number of values:", len(marks))
print("Highest mark:", max(marks))
print("Lowest mark:", min(marks))
print("Total marks:", sum(marks))
print("Average:", sum(marks) / len(marks))
```

---

## 2.7 Tuples

A tuple is similar to a list, but its values cannot be changed after creation.

```python
# Coordinates should remain fixed, so a tuple is appropriate.
coordinates = (30.90, 75.85)

print("Latitude:", coordinates[0])
print("Longitude:", coordinates[1])
```

**List versus tuple:**

| Feature | List | Tuple |
|---|---|---|
| Syntax | `[ ]` | `( )` |
| Changeable | Yes | No |
| Typical use | Data that may change | Fixed data |

---

## 2.8 Dictionaries

A dictionary stores values using meaningful keys.

```python
student = {
    "id": 101,
    "name": "Aman",
    "marks": 82
}

# Access a value through its key.
print("Name:", student["name"])

# Add a new key-value pair.
student["city"] = "Jalandhar"

# Update an existing value.
student["marks"] = 86

print(student)
```

Dictionaries are important because Pandas DataFrames are often created from dictionaries.

---

## 2.9 Loops

Loops repeat a block of code.

### `for` loop

```python
marks = [72, 85, 68, 90]

# Each value from the list is assigned to 'mark' one by one.
for mark in marks:
    print("Current mark:", mark)
```

### Loop with a condition

```python
marks = [72, 35, 68, 29, 90]

for mark in marks:
    if mark >= 40:
        print(mark, "-> Pass")
    else:
        print(mark, "-> Fail")
```

### `while` loop

```python
count = 1

# Continue while the condition remains True.
while count <= 5:
    print("Count:", count)
    count += 1  # Increase count to avoid an infinite loop.
```

---

## 2.10 Functions

A function is a reusable block of code.

```python
def calculate_average(values):
    """Return the arithmetic mean of a list of numbers."""

    # sum(values) adds all numbers.
    # len(values) returns the number of items.
    average = sum(values) / len(values)
    return average

student_marks = [72, 85, 68, 90]
result = calculate_average(student_marks)

print("Average:", result)
```

<table><tr><td bgcolor="#E8F4FD">
<b>💡 Why functions matter in ML</b><br>
Functions help organise repeated tasks such as cleaning columns, calculating metrics, plotting graphs, and training models.
</td></tr></table>

---

# 3. NumPy Arrays

## 3.1 What is NumPy?

**NumPy** stands for Numerical Python. It provides fast, memory-efficient arrays and mathematical operations.

A Python list can store mixed data types, while a NumPy array generally stores values of the same type. This makes numerical computation faster and more predictable.

---

## 3.2 Creating a One-Dimensional Array

```python
import numpy as np

# Convert a Python list into a NumPy array.
marks = np.array([72, 85, 68, 90, 77])

print(marks)
print("Data type:", marks.dtype)
print("Dimensions:", marks.ndim)
print("Shape:", marks.shape)
print("Number of elements:", marks.size)
```

### Understanding array properties

| Property | Meaning |
|---|---|
| `dtype` | Data type of elements |
| `ndim` | Number of dimensions |
| `shape` | Size along each dimension |
| `size` | Total number of elements |

---

## 3.3 Creating Two-Dimensional Arrays

A two-dimensional array is similar to a table containing rows and columns.

```python
student_data = np.array([
    [101, 72, 80],
    [102, 85, 88],
    [103, 68, 75]
])

print(student_data)
print("Dimensions:", student_data.ndim)
print("Shape:", student_data.shape)  # 3 rows and 3 columns
```

---

## 3.4 Special Arrays

```python
# Create an array containing five zeros.
zeros_array = np.zeros(5)

# Create a 2 x 3 array containing ones.
ones_array = np.ones((2, 3))

# Create values from 0 to 8 with a step of 2.
range_array = np.arange(0, 10, 2)

# Create five equally spaced values between 0 and 1.
linear_array = np.linspace(0, 1, 5)

print("Zeros:", zeros_array)
print("Ones:\n", ones_array)
print("Range:", range_array)
print("Equally spaced:", linear_array)
```

---

## 3.5 Indexing and Slicing

```python
values = np.array([10, 20, 30, 40, 50])

print(values[0])      # First element
print(values[-1])     # Last element
print(values[1:4])    # Elements at positions 1, 2, and 3
print(values[:3])     # First three elements
print(values[::2])    # Every second element
```

### Two-dimensional indexing

```python
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Syntax: matrix[row_index, column_index]
print("Row 2, column 3:", matrix[1, 2])

# Select the complete first row.
print("First row:", matrix[0, :])

# Select the complete second column.
print("Second column:", matrix[:, 1])
```

---

## 3.6 Vectorised Operations

NumPy applies an operation to every array element without writing an explicit loop.

```python
marks = np.array([72, 85, 68, 90])

# Add 5 to every value.
bonus_marks = marks + 5

# Convert each mark into a proportion.
proportions = marks / 100

# Square each value.
squared_marks = marks ** 2

print("Bonus marks:", bonus_marks)
print("Proportions:", proportions)
print("Squared values:", squared_marks)
```

<table><tr><td bgcolor="#EAF7EA">
<b>✅ Benefit</b><br>
Vectorised NumPy code is usually shorter and faster than manually looping through every value.
</td></tr></table>

---

## 3.7 Aggregate Functions

Aggregate functions summarise an entire array.

```python
marks = np.array([72, 85, 68, 90, 77])

print("Sum:", np.sum(marks))
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Standard deviation:", np.std(marks))
```

---

## 3.8 Reshaping Arrays

Reshaping changes the arrangement of values without changing the data.

```python
values = np.array([1, 2, 3, 4, 5, 6])

# Convert six values into a matrix of 2 rows and 3 columns.
reshaped = values.reshape(2, 3)

print(reshaped)
```

<table><tr><td bgcolor="#FFF4CE">
<b>⚠️ Important</b><br>
The new shape must contain the same number of elements. Six values can be reshaped to <code>(2, 3)</code> or <code>(3, 2)</code>, but not <code>(4, 2)</code>.
</td></tr></table>

---

## 3.9 Boolean Filtering

```python
marks = np.array([72, 35, 68, 29, 90])

# Create a Boolean mask.
pass_mask = marks >= 40
print("Mask:", pass_mask)

# Keep only values where the mask is True.
passing_marks = marks[pass_mask]
print("Passing marks:", passing_marks)
```

---

## 3.10 Handling Missing Numerical Values

NumPy represents many missing numerical values using `np.nan`.

```python
scores = np.array([72, 85, np.nan, 90, 77])

# np.mean() returns nan when a missing value is present.
print("Regular mean:", np.mean(scores))

# np.nanmean() ignores missing values.
print("Mean ignoring NaN:", np.nanmean(scores))

# Identify missing positions.
print("Missing-value mask:", np.isnan(scores))
```

---

# 4. Mini Practice Program

```python
import numpy as np

# Store marks of five students.
marks = np.array([78, 92, 35, 64, 88])

# Calculate summary information.
average_mark = np.mean(marks)
highest_mark = np.max(marks)
lowest_mark = np.min(marks)

# Filter students who passed.
passing_marks = marks[marks >= 40]

print("All marks:", marks)
print("Average:", average_mark)
print("Highest:", highest_mark)
print("Lowest:", lowest_mark)
print("Passing marks:", passing_marks)
```

---

# 5. Practice Exercises

1. Create variables for a student's name, registration number, CGPA, and placement status. Print each value and its data type.
2. Write a conditional program that displays `Eligible` when CGPA is at least 7.0 and attendance is at least 75.
3. Create a list of ten marks and calculate its minimum, maximum, and average.
4. Write a function that accepts a list of values and returns the number of passing marks.
5. Create a NumPy array containing values from 10 to 100 with a step of 10.
6. Reshape the array into two rows and five columns.
7. Extract values greater than 50 using Boolean filtering.
8. Add 5 bonus marks to every value using a vectorised operation.

---

# 6. Quick Revision

| Concept | Main purpose |
|---|---|
| Variable | Stores a value |
| Condition | Makes a decision |
| Loop | Repeats instructions |
| Function | Creates reusable logic |
| List | Stores an editable collection |
| Dictionary | Stores key–value data |
| NumPy array | Stores numerical data efficiently |
| Boolean mask | Filters values using conditions |
| Reshape | Changes array dimensions |
| Aggregation | Produces summary values |

---

## ➡️ Continue

Proceed to [Part 2: Pandas, Data Loading, and Missing Values](CSE570_Unit_1_Part_2_Pandas_and_Data_Cleaning.md).
