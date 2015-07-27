import pandas as pd
import csv
import numpy as np

df_old = pd.read_csv('burncoat.csv')

skill_col = df_old['skill']
skills = set()
for skill in skill_col:
	if skill not in skills and skill != 'noskill':
		skills.add(skill)
# print(skills,len(skills)) # 98 unique skills

student_col = df_old['name']
students = set()
for student in student_col:
	if student not in students:
		students.add(student)
# print(students, len(students)) # 338 unique students 
df_new = pd.DataFrame(index=list(students))
df_new['student'] = students

for skill in skills:
	df_new[skill + '_count'] = 0
	df_new[skill + '_percent_correct'] = 0

# test = []
# for _ in range(5):
# 	test.append(students.pop())

i = 0
for student in students:
	print(student, i)
	i += 1
	for skill in skills:
		count = len(df_old.loc[(df_old['name'] == student) & (df_old['skill'] == skill), 'name'].values)
		df_new.loc[df_new['student'] == student, skill + '_count'] = count

		if count != 0:
			correct =  sum(df_old.loc[(df_old['name'] == student) & (df_old['skill'] == skill) & (df_old['correct'] == 1), 'correct'].values)
			df_new.loc[df_new['student'] == student, skill + '_percent_correct'] = float(correct) / count 

df_new = df_new.drop('student', 1)

for skill in skills:
	if df_new[skill + '_count'].sum() < 60:
		df_new = df_new.drop(skill + '_count', 1)
		df_new = df_new.drop(skill + '_percent_correct', 1)

df_new.to_csv('burn_without_60.csv', index = False, header = False)

# f = open('burn_new.csv', 'wb')
# w = csv.writer(f)
# w.writerow(['student'] + [skill + '_count' for skill in skills]) # + [skill + '_percent_correct' for skill in skills]

# for student in test:
# 	print(student, i )
# 	i -= 1
# 	for skill in skills:
# 		output = [student]
# 		output += df_new[skill + '_count'][df_new['student'] == student].values
# 		print(len(output))
# 		# print(df[skill + '_percent_correct'][df['name'] == student].values)
# 	w.writerow(output)


# f.close()