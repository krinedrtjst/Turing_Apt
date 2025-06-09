#!/usr/bin/env python
# coding: utf-8

# ## Valid Parentheses:
# Given a string s containing just the characters '(', ')', '[', ']', '{', '}', determine if the input string is valid.
# 
# An input string is valid if:
# 
# Open brackets must be closed by the same type of brackets
# Open brackets must be closed in the correct order
# Constraints
# 1 <= s.lenght <= 104
# s consists of parentheses only '()[]{}'
# Example 1:
# 
# Input: s = "()"
# 
# Output: valid
# 
# Example 2:
# 
# Input: s = "()[]{}"
# 
# Output: valid
# 
# Example 3:
# 
# Input: s = "(]"
# 
# Output: invalid
# 
# Example 4:
# 
# Input: s = "([)]"
# 
# Output: invalid
# 
# Example 5:
# 
# Input: s = "{[]}"
# 
# Output: valid

# In[2]:


def isValid(s: str) -> bool:
    #Create a stack to keep track of opening brackets 
    stack = []
    #Dictionary to map closing brackets to their corresponding opening brackets
    brackets_map =s

    #Iterate through each caracter in the string 
    for char in s:
        #If it's a closing bracket 
        if char in brackets_map:
            # if stack is empty but we have a closing bracket, it's invalid
            if not stack:
                return False
            #Pop the last opening bracket, push it onto the stack
            last_opening = stack.pop()
            if last_opening != brackets_map[char]:
                return False
            #If it's an opening bracket, push it onto the stack
            else: 
                stack.append(char)

    #After processing all characters, stack should be empty for a valid string
    return len(stack) == 0
#Test cases
test_cases = ["()","()[]{}","(]","[[)]","{[]}"]
for test in test_cases:
    result = "valid" if isValid(test) else "invalid"
    print(f"Input: s = \"{test}\"")
    print(f"Output: {result}\n")
            
    


# #Let's go through how this solution works:
#  - We use a stack to keep track of opening brackets ('(', '[', '{').
# - We create a dictionary brackets_map that maps closing brackets to their corresponding opening brackets for easy lookup.
# - For each character in the input string:
# - If it's a closing bracket (')', ']', '}'):
# - Check if stack is empty (if yes, return False as we have a closing bracket without an opening one)
# - Pop the last opening bracket from stack and verify it matches the current closing bracket
# - If it doesn't match, return False
# - If it's an opening bracket, push it onto the stack
# - Finally, check if the stack is empty. If it is, all brackets were properly closed (return True). If not, we have unclosed opening brackets (return False).
# - Explanation of Results
# Example 1: "()" - Valid because the opening bracket is closed by the same type.
# Example 2: "()[]{}" - Valid because all brackets are properly nested and matched.
# Example 3: "(]" - Invalid because the opening parenthesis is closed by a square bracket (wrong type).
# Example 4: "([)]" - Invalid because the brackets are not closed in the correct order (nested brackets aren't properly matched).
# Example 5: "{[]}" - Valid because the brackets are properly nested and matched.
# Optimizations and Notes
# The solution is already highly efficient for the given constraints, with linear time and space complexity.
# The use of a dictionary (brackets_map) for matching brackets is faster than multiple conditional checks.
# The code is readable and maintainable, with clear variable names and comments.
# Since the input is constrained to contain only the characters '()[]{}', no additional input validation is needed.
# 
# 

# # Baseball Game:
# You are keeping score for a baseball game with strange rules. The game consists of several rounds where the scores of past rounds may affect future rounds' scores.
# 
# At the beginning of the game, you start with an empty record. you are given a list of strings ops, where ops[i] is the ith operation you must apply to the record and is one of the following:
# 
# An integer x - Record a new score of X.
# "+" - Record a new score that is the sum of the previous two scores. It is guaranteed that there will always be two previous scores.
# "D" - Record a new score that is double the previous score. It is guaranteed that there will always be a previous score.
# "C" - Invalidate the previous score, removing it from the record. It is guaranteed that there will always be a previous score.
# Return the sum of all the scores on the record.
# 
# Example 1:
# 
# Input: ops = ["5", "2", "C", "D", "+"]
# 
# Output: 30
# 
# Example 2:
# 
# Input: ops = ["5", "-2", "4", "C", "D", "9", "+", "+"]
# 
# Output: 27
# 
# Example 3:
# 
# Input: ops = ["1"]
# 
# Output: 1
# 
# Constraints
# 1 <= ops.lenght <= 1000
# ops[i] is "C", "D", "+", or a string representing an integer in the range [-3 * 104, 3 * 104]
# For operation "+", there will always be two previous scores on the record.
# For operation "C" and "D", there will always be at least one previous score on the record.

# In[5]:


def calPoints(ops: list[str]) -> int: 
    # Initialize an empty list to store the record of scores
    record = []
    # Process each operation in the input list 
    for op in ops:
        if op == "C":
            # Cancel the last score by removing it from the record 
            record.pop()
        elif op == "D":
           #Double the last score and add it to the record
            record.append(record[-1]*2)
        elif op == "+":
            # Sum the last two scores and add the result to the record 
            record.append(record[-1] + record [-2])
        else: 
            # If the operation is a number, convert it to int and add to record 
            record.append(int(op))
    # Return the sum of all scores in the record 
    return sum(record)
# Test cases 
test_cases = [
    ["5","2","C","D","+"],
    ["5","-2","4","C","D","9","+","+"],
    ["1"]     
]
for test in test_cases:
    result = calPoints(test)
    print(f"Input: ops = {test}")
    print(f"Output: {result}\n")
            


# #We use a list (record) to keep track of the scores as they are added or modified.
# - For each operation in the input list ops:
# If it's "C", remove the last score using pop().
# If it's "D", double the last score (record[-1] * 2) and append it.
# If it's "+", sum the last two scores (record[-1] + record[-2]) and append the result.
# If it's a number (as a string), convert it to an integer using int() and append it.
# Finally, return the sum of all scores in the record list using sum().
# Constraints Handling:
# The solution handles the length constraint (1 <= ops.length <= 1000) naturally as it processes each operation sequentially.
# It supports integer inputs in the range [-3 * 10^4, 3 * 10^4] since Python integers have no practical upper limit in this context.
# As per the problem, operations "C", "D", and "+" are guaranteed to have the necessary previous scores, so no additional checks are needed for empty or insufficient records.
# Time and Space Complexity:
# Time Complexity: O(n), where n is the length of the input list ops, as we process each operation once. The sum() at the end is also O(n) in the worst case, but it's a single pass.
# Space Complexity: O(n), for storing the record list, which can grow up to the size of ops in the worst case (if no "C" operations are present).

# #The task involves calculating how many flies each frog can eat based on their positions and tongue sizes. Here's a detailed breakdown of the problem:
# Input:
# X: A list of integers representing the positions of frogs on a line.
# S: A list of integers representing the tongue sizes of the corresponding frogs.
# Y: A list of integers representing the positions of flies on the line.
# Logic:
# A frog at position X[i] can eat a fly at position Y[j] if the absolute difference between their positions (|X[i] - Y[j]|) is less than or equal to the frog's tongue size S[i].
# For each frog, you need to count how many flies it can eat based on this condition.
# Output:
# The function should return a list of integers where the i-th integer represents the number of flies the i-th frog can eat.
# Examples:
# Example 1:
# Input: X = [1, 4, 5], S = [2, 3, 5], Y = [2, 3, 5]
# Output: [2, 3, 3]
# Explanation:
# Frog at X[0] = 1 with S[0] = 2 can eat flies at Y[0] = 2 (distance 1) and Y[1] = 3 (distance 2), so 2 flies.
# Frog at X[1] = 4 with S[1] = 3 can eat flies at Y[0] = 2 (distance 2), Y[1] = 3 (distance 1), and Y[2] = 5 (distance 1), so 3 flies.
# Frog at X[2] = 5 with S[2] = 5 can eat all flies (distances 3, 2, 0), so 3 flies.
# Example 2:
# Input: X = [3], S = [5], Y = [1, 2]
# Output: [2]
# Explanation: Frog at X[0] = 3 with S[0] = 5 can eat flies at Y[0] = 1 (distance 2) and Y[1] = 2 (distance 1), so 2 flies.
# Constraints:
# The input lists X, S, and Y are of the same length, implying one frog per position and tongue size.
# The problem assumes the input is well-formed (no need to handle invalid cases like mismatched lengths).

# In[ ]:


from typing import List 

def frogs (X:List [int], S:List[int], Y: List[int]) -> List [int]:
    #Initialize result list to store the number of flies each frog can eat
    result = []

    # Iterate over each frog
    for i in range(len(x)):
        frog_pos = x [i] #position of i-th frog
        tongue_size = S[i] #Tongue size of the i-th frog 
        fly_count = o #counter for flies this frog can eat
    # Check each fly's position
    for fly_pos in Y:
        #Calculate absolute distance between frog and fly
        distance = abs(frog_pos - fly_pos)
        #If distance is <= tongue size, frog can eat the fly
        if distance <= tongue_size:
            fly_count += 1
    #Add the count to result
    result.append(fly_count)
    return result 
# Read input from standart input (Turing Environment)
if __name__== "__main__":
    #Do not change the code below, we use it to grade your submission.If changed the input won't work
   line = input ()
   x = [int(i) for i in line.strip().split()]
   line = input ()
   s = [int(i) for i in line.strip().split()]
   line = input ()
   y = [int(i) for i in line.strip().split()]
   #Calculate and print result
   result = frogs (x , s, y)
   print(*result) #unpack list for output


# In[ ]:


a = input().strip() #read string a
t = input().strip() #read string t
out = ''

if len(t) == 1
   out = t 
else: 
    c = 0
    for i in range(min(len(a), len(t)))):
        if a[i] == t[i]:
            c += 1
        else: 
            out = t[i]
            break
    if not out and c == len (a):
        out = t[c + 1] if c + 1 < len(t) else t [-1]
    elif not out:
        out = t[-1]
print(out)


# # Logic Implementation:
# - For each frog at position X[i] with tongue size S[i], iterate over all fly positions in Y.
# Calculate the absolute distance (abs(frog_pos - fly_pos)).
# Increment fly_count if the distance is less than or equal to the tongue size.
# Append the fly_count to the result list.
# Input Handling:
# The code reads three lines of input (for X, S, and Y) as per the Turing environment’s format.
# Each line is split into integers and stored in the respective lists.
# The if __name__ == "__main__": block is duplicated in the original, but we keep the second instance as per the "DO NOT CHANGE" instruction, assuming it’s part of the grading setup.
# Output Format:
# The result is printed with print(*result), which unpacks the list into space-separated integers (e.g., 2 3 3), matching the expected output format.
# Efficiency:
# The solution has a time complexity of O(n*m), where n is the number of frogs and m is the number of flies. For small inputs (as in the examples), this is acceptable.
# No optimization (e.g., sorting or binary search) is needed given the problem constraints.
# Accuracy:
# The logic correctly handles the examples:
# Example 1: [2, 3, 3] matches the expected output.
# Example 2: [2] matches the expected output.
# Testing the Solution
# Test Case 1 (from image):
# Input: (assumed from console output 1 4 5, 2 3 5, 2 3 5)
# Expected Output: 2 3 3
# Result: Matches the calculation above.
# Test Case 2:
# Input: 3, 5, 1 2
# Expected Output: 2
# Result: Matches the calculation.
# Additional Considerations
# Edge Cases:
# If X, S, or Y is empty, the function returns an empty list, which is valid.
# If a frog’s position equals a fly’s position (distance = 0), it can eat the fly (included in <= condition).
# Multiline Input: The code assumes one line per list, as shown in the Turing interface.
# Language: The problem is in English, so no language-specific OCR is needed here (unlike the previous PDF context).
# Final Output
# When run in the Turing environment with the provided input format, the code will produce the correct output (e.g., 2 3 3 for Test Case 1). The solution adheres to the problem’s requirements and preserves the original code structure below line 12 as instructed.
# 
# If you need to test this locally or integrate it with OCR for scanned images (as per your earlier context), let me know, and I can adapt the code accordingly!

# In[ ]:


get_ipython().system('pip install seaborn matplotlib')


# 

# The provided Python code processes health and COVID-19 datasets using pandas for various analytical tasks (Q1-Q12). While the code is functional, it can be optimized for performance, memory usage, and readability by leveraging vectorized operations, reducing redundant computations, and applying best practices. Below is an optimized version with detailed explanations of the improvements.

# Let’s conduct a detailed data analysis of the provided Python code, which processes health-related datasets (cardio_alco.csv, cardio_base.csv) and COVID-19 data (covid_data.csv) to derive various insights. The analysis will cover the data loading, preprocessing, and each query (Q1-Q12), interpreting the potential findings based on typical dataset characteristics. Since the actual data isn’t provided, I’ll base the analysis on the code’s intent, common patterns in such datasets, and logical inferences as of 04:35 PM -03 on Sunday, June 08, 2025.
# 
# 1. Data Loading and Preprocessing
# Datasets:
# cardio_alco.csv: Likely contains alcohol consumption data linked by id, with delimiter=";" indicating a semicolon-separated format.
# cardio_base.csv: Contains cardiovascular health data (e.g., age, weight, height, cholesterol, gender, smoke, ap_hi - systolic blood pressure).
# covid_data.csv: Contains COVID-19 statistics (e.g., new_cases, new_deaths, population, hospital_beds_per_thousand, gdp_per_capita).
# Optimization:
# Dtypes: Specified types (e.g., int32 for id, float32 for weight) reduce memory usage. Nullable types (Int32, Int64) handle NA values in covid_data, which is critical for integer columns with missing data.
# Low Memory: low_memory=True processes files in chunks, suitable for large datasets.
# Age Preprocessing: df_cardio_base["age"] = (df_cardio_base["age"] / 365).astype('int32') converts age from days to years, assuming the original data is in days (a common format in health datasets).
# Potential Insights:
# The dataset sizes and NA prevalence can be checked with df_cardio_base.info() or df_covid.isna().sum(). Missing values in new_cases or new_deaths might indicate unreported data, affecting aggregates.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data with optimized parameters
data_files = {
    "cardio_alco": r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\cardio_alco.csv",
    "cardio_base": r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\cardio_base.csv",
    "covid_data": r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\covid_data.csv"
}

df_cardio_alco = pd.read_csv(data_files["cardio_alco"], delimiter=";", dtype={'id': 'int32'}, low_memory=True)
df_cardio_base = pd.read_csv(data_files["cardio_base"], dtype={'id': 'int32', 'age': 'int32', 'weight': 'float32', 'height': 'float32', 'cholesterol': 'int8', 'gender': 'int8', 'smoke':


# In[ ]:


# Load data with optimized parameters
data_files = {
    "cardio_alco": r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\cardio_alco.csv",
    "cardio_base": r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\cardio_base.csv",
    "covid_data": r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\covid_data.csv"
}

df_cardio_alco = pd.read_csv(data_files["cardio_alco"], delimiter=";", dtype={'id': 'int32'}, low_memory=True)
df_cardio_base = pd.read_csv(data_files["cardio_base"], dtype={'id': 'int32', 'age': 'int32', 'weight': 'float32', 'height': 'float32', 'cholesterol': 'int8', 'gender': 'int8', 'smoke': 'int8', 'ap_hi': 'int32'}, low_memory=True)
df_covid = pd.read_csv(data_files["covid_data"], dtype={
    'new_cases': 'Int32',  # Nullable integer to handle NA
    'new_deaths': 'Int32',  # Nullable integer to handle NA
    'population': 'Int64',  # Nullable integer for large populations
    'hospital_beds_per_thousand': 'float32',
    'gdp_per_capita': 'float32'
}, low_memory=True)


# In[ ]:


# Pre-compute age in years once
df_cardio_base["age"] = (df_cardio_base["age"] / 365).astype('int32')


# In[ ]:


# Q1: Average weight by age group with optimized grouping
q1 = df_cardio_base.groupby("age", as_index=False)["weight"].agg("mean").astype('float32')


# 1.Q1: Average Weight by Age Group
# Analysis: Computes the mean weight for each age, providing a profile of weight distribution across ages.
# Insight: Younger individuals might have lower average weights, with a potential increase or stabilization in middle age, followed by a decline in older age due to health issues. Outliers (e.g., very high weights) could skew results, detectable via q1.sort_values(by="weight").

# In[ ]:


# Q2: Mean cholesterol for age > 50 and <= 50 using vectorized conditions
# Add age_group column to avoid FutureWarning
df_cardio_base["age_group"] = (df_cardio_base["age"] > 50).astype('int8')
q2 = df_cardio_base.groupby("age_group", as_index=False)["cholesterol"].mean().rename(columns={"age_group": "age_group"})
# Optionally drop the temporary column if not needed further
df_cardio_base.drop(columns=["age_group"], inplace=True)


# In[ ]:


Analysis: Compares average cholesterol levels between age groups (0 = <= 50, 1 = > 50). Cholesterol levels typically rise with age due to metabolic changes.
Insight: If q2.loc[q2["age_group"] == 1, "cholesterol"] > q2.loc[q2["age_group"] == 0, "cholesterol"], it supports the hypothesis of age-related cholesterol increase. Check for NA values with df_cardio_base["cholesterol"].isna().sum().


# In[ ]:


# Q3: Total smokers by gender using vectorized sum
q3 = df_cardio_base.groupby("gender")["smoke"].sum().reindex([1, 2], fill_value=0).astype('int32')


# Analysis: Sums the smoke column (binary: 1 = smoker, 0 = non-smoker) by gender (assuming 1 = male, 2 = female).
# Insight: If q3[1] > q3[2], males smoke more, aligning with global trends. The reindex([1, 2]) ensures both genders are represented, even if data is missing for one.

# In[ ]:


# Q4: Top 1% of heights using numpy percentile for efficiency
height_quantile = np.percentile(df_cardio_base["height"], 99)
q4 = df_cardio_base.loc[df_cardio_base["height"] >= height_quantile, "height"]


# Identifies individuals in the top 1% of heights, useful for detecting outliers or unusual data entries.
# Insight: Compare q4.mean() with the overall df_cardio_base["height"].mean() to assess if the top 1% is significantly taller (e.g., > 190 cm might indicate errors or elite athletes). Check for implausible values (e.g., > 250 cm) with q4.max().

# In[ ]:


# Q5: Spearman correlation matrix with optimized computation
q5 = df_cardio_base.select_dtypes(include=[np.number]).corr(method="spearman")
plt.figure(figsize=(20, 10))
sns.heatmap(q5, annot=True)
plt.show()


# In[ ]:


Computes Spearman rank correlation (non-parametric, robust to outliers) for numeric columns and visualizes it.
Insight:
Strong positive correlations (e.g., age vs. cholesterol > 0.5) suggest age-related health risks.
Negative correlations (e.g., height vs. weight < -0.3) might indicate body composition differences.
The heatmap’s annotations (annot=True) highlight significant values. Check for multicollinearity (correlations > 0.8) that might affect models.


# In[ ]:


# Q6: Height statistics and outlier filtering with single pass
height_stats = df_cardio_base["height"].agg(["mean", "std"]).astype('float32')
q6a, q6b = height_stats["mean"], height_stats["std"]
q6c = df_cardio_base.loc[df_cardio_base["height"] > (q6a + 2 * q6b)]


# Calculates mean and standard deviation of height, then filters outliers (> mean + 2*std, approximately 95th percentile).
# Insight: q6c might reveal data entry errors (e.g., heights > 200 cm) or rare tall individuals. Compare q6c.shape[0] to total rows to estimate outlier prevalence.

# In[ ]:


# Q7: Merge and filter for age > 50 with alcohol consumption
q7 = pd.merge(df_cardio_base, df_cardio_alco, on="id", how="inner")
q7a = q7.loc[(q7["age"] > 50) & (q7["alco"] == 1)].copy()
q7b = q7.loc[q7["age"] > 50].copy()


# Merges datasets on id and filters for individuals over 50, with q7a focusing on alcohol consumers (alco == 1).
# Insight: Compare q7a.shape[0] vs. q7b.shape[0] to estimate alcohol consumption prevalence among those over 50. Analyze health metrics (e.g., q7a["cholesterol"].mean()) for potential alcohol-related effects.

# In[ ]:


# Q8: Systolic blood pressure for smokers vs. non-smokers
q8 = df_cardio_base.groupby("smoke")["ap_hi"].mean().reindex([1, 0], fill_value=0).astype('int32')


# Computes average systolic blood pressure (ap_hi) for smokers (1) and non-smokers (0).
# Insight: If q8[1] > q8[0], it suggests smoking increases blood pressure, a known cardiovascular risk factor. Check statistical significance with a t-test if sample sizes allow.

# In[ ]:


# Q9: Sum of new cases by date for Germany and Italy
q9_mask = df_covid["location"].isin(["Germany", "Italy"])
q9a = df_covid.loc[q9_mask].groupby("date")["new_cases"].sum().reset_index().astype({'new_cases': 'Int32'})


# Aggregates daily new COVID-19 cases for Germany and Italy.
# Insight: Plot q9a to identify peak dates (e.g., March 2020). Compare total cases (q9a["new_cases"].sum()) between countries.

# In[ ]:


# Q10: New cases in Italy between specific dates
q10_mask = (df_covid["location"] == "Italy") & (df_covid["date"].between("2020-02-28", "2020-03-20"))
q10a = df_covid.loc[q10_mask].groupby("date")["new_cases"].sum().reset_index().astype({'new_cases': 'Int32'})


# Insight: q10a["new_cases"].sum() gives total cases in this period. A rising trend might reflect the initial outbreak’s spread.

# In[ ]:


# Q11: Death-to-population ratio by location
q11a = df_covid.groupby("location")["new_deaths"].sum()
q11b = df_covid.groupby("location")["population"].mean()
q11 = pd.DataFrame({"population": q11b, "deaths": q11a, "ratio": q11a / q11b}).fillna(0).astype({'deaths': 'Int32', 'ratio': 'float32'})


# Calculates the ratio of total deaths to average population by location.
# Insight: High ratio values (e.g., > 0.001) indicate severe impact. Compare with hospital_beds_per_thousand to assess healthcare capacity.

# In[ ]:


# Q12: Hospital beds and GDP per capita correlation by location
q12 = df_covid.groupby("location").agg({"hospital_beds_per_thousand": "mean", "gdp_per_capita": "mean"}).astype({'hospital_beds_per_thousand': 'float32', 'gdp_per_capita': 'float32'})
q12c = q12.corr().loc["hospital_beds_per_thousand", "gdp_per_capita"]


# Computes the correlation between average hospital beds per thousand and GDP per capita.
# Insight: A positive q12c (e.g., > 0.5) suggests wealthier regions have better healthcare infrastructure. A low or negative value might indicate disparities.

# Health Trends: Q1, Q2, Q6, and Q8 suggest age and lifestyle (smoking, weight) impact cardiovascular health. Cross-reference with q7a to study alcohol’s role.
# COVID-19 Impact: Q9-Q11 highlight pandemic severity, with Q12 linking healthcare capacity to economic factors.
# Outliers and Errors: Q4 and Q6c are critical for data cleaning. Investigate q6c heights with df_cardio_base.loc[q6c.index] to confirm validity.
# Visualization: Q5’s heatmap is a powerful tool; focus on clusters (e.g., age vs. cholesterol) for hypothesis generation.

# # vacation rental listings and reviews

# In[40]:


import pandas as pd
import os
import sys

# Specify the file path
file_path = r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\AB_NYC_2019.csv\AB_NYC_2019.csv"

# Function to check and handle file access
def read_csv_with_permission(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist. Please verify the path.")
        return None
    
    # Attempt to read the file with error handling
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        print("File read successfully!")
        print(df.head())  # Display the first few rows to verify
        return df
    except PermissionError:
        print(f"PermissionError: You do not have permission to access {file_path}.")
        print("Suggestions:")
        print("- Run this script or your IDE as an administrator.")
        print("- Check file permissions: Right-click the file > Properties > Security > Ensure 'Read' permission for your user.")
        print("- Move the file to a local directory (e.g., C:\\Users\\lost4\\Documents\\airbnb.csv) and update the path.")
        return None
    except FileNotFoundError:
        print(f"FileNotFoundError: The file {file_path} was not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Execute the function
if __name__ == "__main__":
    df = read_csv_with_permission(file_path)
    if df is not None:
        # Optional: Save to a new location if needed
        df.to_csv(r"C:\Users\lost4\Documents\airbnb_local.csv", index=False)
        print("Data saved to C:\\Users\\lost4\\Documents\\airbnb_local.csv")


# In[62]:


import pandas as pd


# Write your code here
def neighborhood_with_highest_median_price_diff(df_listings: pd.DataFrame) -> str:
    # Group by neighborhood and calculate median price for superhosts and non-superhosts
    superhost_prices = df_listings[df_listings['host_is_superhost'] == True].groupby('neighbourhood_cleansed')['price'].median()
    non_superhost_prices = df_listings[df_listings['host_is_superhost'] == False].groupby('neighbourhood_cleansed')['price'].median()
    
    # Merge and calculate the difference
    price_diff = (superhost_prices - non_superhost_prices).fillna(0)
    
    # Find the neighborhood with the highest price difference
    return price_diff.idxmax() if not price_diff.empty else ""

# MANDATORY - Explain your solution in plain english here
# This code finds the neighborhood where the difference in median price between superhosts and non-superhosts is the largest. It first separates the listings into two groups based on whether the host is a superhost, calculates the median price for each neighborhood in both groups, then finds the difference. The neighborhood with the biggest difference is returned.

#COMMENTS END
...

if __name__ == '__main__':
    print('Neighborhood with highest price difference')


# In[64]:


import pandas as pd


# Write your code here
def review_score_with_highest_correlation_to_price(df_listings: pd.DataFrame) -> str:
    # Calculate correlation of price with each review score column
    correlations = df_listings[['price', 'review_scores_rating', 'review_scores_accuracy', 
                               'review_scores_cleanliness', 'review_scores_checkin', 
                               'review_scores_communication', 'review_scores_location', 
                               'review_scores_value']].corr()['price'].drop('price')
    
    # Find the review score with the highest absolute correlation
    return correlations.abs().idxmax() if not correlations.empty else ""

# MANDATORY - Explain your solution in plain english here
# This code checks how strongly each review score (like rating, accuracy, cleanliness, etc.) relates to the price of a listing. It uses a statistical method to measure this relationship for all review score columns and picks the one with the strongest connection to price.

#COMMENTS END
...

if __name__ == '__main__':
    print('Review score with max correlation to price')


# In[106]:


import pandas as pd
import os

# Specify the file path
file_path = r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\AB_NYC_2019.csv"

# Write your code here
def prof_nonprof_host_price_diff(df_listings: pd.DataFrame) -> float:
    # Identify professional hosts (more than 5 unique locations)
    professional_hosts = df_listings.groupby('host_id')['neighbourhood'].nunique() > 5
    professional_hosts = professional_hosts[professional_hosts].index
    
    # Calculate average price for professional and non-professional hosts
    prof_price = df_listings[df_listings['host_id'].isin(professional_hosts)]['price'].mean()
    non_prof_price = df_listings[~df_listings['host_id'].isin(professional_hosts)]['price'].mean()
    
    # Return the difference
    return prof_price - non_prof_price if pd.notna([prof_price, non_prof_price]).all() else 0.00


# In[108]:


import pandas as pd
import os

# Specify the file path
file_path = r"C:\Users\lost4\OneDrive\Documentos\DATA\job related\turing data set\AB_NYC_2019.csv"

# Write your code here
def price_premium_for_entire_homes(df_listings: pd.DataFrame) -> float:
    # Filter for entire homes/apartments (assuming 'room_type' column exists)
    entire_homes = df_listings[df_listings['room_type'] == 'Entire home/apt']['price'].median()
    other_listings = df_listings[df_listings['room_type'] != 'Entire home/apt']['price'].median()
    
    # Calculate premium
    return entire_homes - other_listings if pd.notna([entire_homes, other_listings]).all() else 0.00

# MANDATORY - Explain your solution in plain english here
# This code calculates the extra median price that entire homes or apartments get compared to other types of listings. It finds the median price for entire homes and for other listing types, then subtracts the other median from the entire homes median to get the premium.


# In[110]:


import pandas as pd


# Write your code here
def listing_with_best_expected_revenue(df_listings: pd.DataFrame) -> int:
    # Placeholder: Assuming revenue is proportional to price and number of reviews
    df_listings['expected_revenue'] = df_listings['price'] * df_listings['number_of_reviews']
    return df_listings.loc[df_listings['expected_revenue'].idxmax(), 'id'] if not df_listings.empty else -1

# MANDATORY - Explain your solution in plain english here
# This code estimates which listing might earn the most by multiplying the price by the number of reviews to get an expected revenue. It then finds the listing ID with the highest calculated revenue.


...

if __name__ == '__main__':
    print('Listing with best expected revenue is:')


# In[112]:


import pandas as pd


# Write your code here
def average_diff_superhost_nonsuperhost_review_score(df_listings: pd.DataFrame) -> float:
    # Calculate average review score for superhosts and non-superhosts
    superhost_score = df_listings[df_listings['host_is_superhost'] == True]['review_scores_rating'].mean()
    nonsuperhost_score = df_listings[df_listings['host_is_superhost'] == False]['review_scores_rating'].mean()
    
    # Return the difference
    return superhost_score - nonsuperhost_score if pd.notna([superhost_score, nonsuperhost_score]).all() else 0.00

# MANDATORY - Explain your solution in plain english here
# This code finds the average difference in review scores between listings managed by superhosts and non-superhosts. It calculates the average review score for each group and subtracts the non-superhost average from the superhost average.


...

if __name__ == '__main__':
    print('Average difference in review scores betwe')


# In[116]:


import pandas as pd


# Write your code here
def host_attribute_with_second_highest_correlation(df_listings: pd.DataFrame) -> str:
    # Calculate correlation of number_of_reviews with host attributes
    correlations = df_listings[['number_of_reviews', 'host_since', 'host_listings_count', 
                               'host_identity_verified', 'calculated_host_listings_count', 
                               'host_is_superhost']].corr()['number_of_reviews'].drop('number_of_reviews')
    
    # Sort by absolute correlation and get the second highest
    sorted_corrs = correlations.abs().sort_values(ascending=False)
    return sorted_corrs.index[1] if len(sorted_corrs) > 1 else ""

# MANDATORY - Explain your solution in plain english here
# This code determines which host-related detail (like how long they’ve been a host or number of listings) has the second-strongest connection to the number of reviews. It measures the relationship between the number of reviews and each host attribute, sorts them by strength, and picks the second one.


...

if __name__ == '__main__':
    print('Host attribute with second highest correlation')

