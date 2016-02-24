import pandas as pd
import csv
import numpy as np

data = pd.read_csv('problem_sample.csv')

# Creating a set for all excercises
skill_col = data['problem_exercise']
skills = set()
for skill in skill_col:
    if skill not in skills:
        skills.add(skill)

# print(skills,len(skills)) # 75 unique skills

# Creatin a row for each student attempt -> USER, all SKILLs (1 if attempted), CORRECT

student_col = data['USER']
i = 0
j = 0

data['correct'] = 0

for skill in skills:
    data[skill] = 0

for row in data.itertuples():
    for skill in skills:
        if row[5] == skill:
            j += 1
            data.loc[data['USER'] == row[1], skill] = 1
    if row[2]:
        i+= 1
        data.loc[data['USER'] == row[1], 'correct'] = 1

# df_new = pd.DataFrame(index=list(students))

df_new = data.drop(['problem_correct','problem_backup_timestamp','problem_time_taken','problem_exercise', \
					'problem_problem_type','problem_seed', 'problem_sha1','problem_hint_used','attempts'], 1)

# index = False, header = False
df_new.to_csv('khan_clean.csv')
