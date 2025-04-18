import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells to the notebook
cells = [
    nbf.v4.new_code_cell("""import os
import sys

# Set Spark environment variables
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize findspark
import findspark
findspark.init()

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder \\
    .appName("nyc-jobs-analysis") \\
    .config("spark.executor.memory", "1g") \\
    .config("spark.driver.memory", "1g") \\
    .getOrCreate()"""),

    nbf.v4.new_markdown_cell("## Data Loading and Initial Exploration"),

    nbf.v4.new_code_cell("""# Read the dataset
df = spark.read.csv("/dataset/nyc-jobs.csv", header=True, inferSchema=True)

# Convert numeric columns
df = df.withColumn("# Of Positions", col("# Of Positions").cast("integer")) \\
       .withColumn("Salary Range From", col("Salary Range From").cast("double")) \\
       .withColumn("Salary Range To", col("Salary Range To").cast("double")) \\
       .withColumn("Posting Date", to_timestamp(col("Posting Date"), "yyyy-MM-dd'T'HH:mm:ss.SSS"))

# Display basic statistics
print("Dataset Overview:")
print(f"Total number of records: {df.count()}")
print(f"Number of columns: {len(df.columns)}")
print("\\nColumn Types:")
df.printSchema()

# Show sample data
df.show(5, truncate=False)"""),

    nbf.v4.new_markdown_cell("## Key Analysis Questions"),

    nbf.v4.new_markdown_cell("### 1. Highest Salary Job Posting per Agency"),
    nbf.v4.new_code_cell("""# Group by Agency and Business Title to find highest salary jobs
highest_salary_jobs = df.groupBy("Agency", "Business Title") \\
                       .agg(
                           max("Salary Range To").alias("Highest Salary"),
                           first("Posting Date").alias("Posting Date"),
                           first("Job Description").alias("Job Description")
                       ) \\
                       .orderBy(col("Highest Salary").desc())

print("Top 10 Highest Paying Jobs by Agency:")
highest_salary_jobs.show(10, truncate=False)"""),

    nbf.v4.new_markdown_cell("### 2. Average Salary per Agency (Last 2 Years)"),
    nbf.v4.new_code_cell("""# Calculate average salaries for the last 2 years
# First, ensure Posting Date is in proper date format
df = df.withColumn("Posting Date", to_timestamp(col("Posting Date"), "yyyy-MM-dd'T'HH:mm:ss.SSS"))

# Get the date two years ago from the most recent posting
max_date = df.select(max("Posting Date")).collect()[0][0]
two_years_ago = df.select(add_months(lit(max_date), -24)).collect()[0][0]  # Subtract 24 months from max date

# Filter and calculate averages
recent_salaries = df.filter(col("Posting Date") >= two_years_ago) \\
                   .groupBy("Agency") \\
                   .agg(
                       avg("Salary Range From").alias("Avg Salary From"),
                       avg("Salary Range To").alias("Avg Salary To"),
                       count(lit(1)).alias("Number of Postings")
                   ) \\
                   .orderBy(col("Avg Salary To").desc())

print("Average Salary per Agency (Last 2 Years):")
print(f"Analysis period: {two_years_ago} to {max_date}")
recent_salaries.show(truncate=False)"""),

    nbf.v4.new_markdown_cell("### 3. Highest Paid Skills Analysis"),
    nbf.v4.new_code_cell("""# Function to extract skills and their associated salaries
def extract_skills(row):
    skills = str(row['Preferred Skills']).lower().split(',')
    return [(skill.strip(), float(row['Salary Range To'])) for skill in skills]

# Process skills and calculate average salaries
skills_rdd = df.select('Preferred Skills', 'Salary Range To').rdd
skills_salary = skills_rdd.flatMap(extract_skills) \\
                         .map(lambda x: (x[0], (x[1], 1))) \\
                         .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \\
                         .map(lambda x: (x[0], x[1][0] / x[1][1], x[1][1])) \\
                         .filter(lambda x: x[2] >= 5) \\
                         .sortBy(lambda x: x[1], ascending=False)

# Display results
print("Top 20 Highest Paid Skills:")
print("Skill | Average Salary | Number of Appearances")
print("-" * 50)
for skill, avg_salary, count in skills_salary.take(20):
    print(f"{skill:30} | ${avg_salary:,.2f} | {count}")"""),

    nbf.v4.new_markdown_cell("### 4. Number of Job Postings per Category (Top 10)"),
    nbf.v4.new_code_cell("""# Count job postings per category
category_counts = df.groupBy("Job Category") \\
                   .count() \\
                   .orderBy(col("count").desc()) \\
                   .limit(10)

print("Top 10 Job Categories by Number of Postings:")
category_counts.show(truncate=False)"""),

    nbf.v4.new_markdown_cell("## Data Visualization"),

    nbf.v4.new_code_cell("""# Convert Spark DataFrames to Pandas for visualization
recent_salaries_pd = recent_salaries.toPandas()
category_counts_pd = category_counts.toPandas()

# Set up the visualization style
plt.style.use('seaborn')

# Create a figure with multiple subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# 1. Average Salary by Agency (Last 2 Years)
recent_salaries_pd = recent_salaries_pd.head(15)  # Top 15 agencies
recent_salaries_pd.plot(kind='barh', x='Agency', y=['Avg Salary From', 'Avg Salary To'], ax=ax1)
ax1.set_title('Average Salary Range by Agency (Last 2 Years)')
ax1.set_xlabel('Average Salary ($)')

# 2. Number of Postings vs Average Salary
ax2.scatter(recent_salaries_pd['Number of Postings'], 
           recent_salaries_pd['Avg Salary To'],
           alpha=0.6)
ax2.set_title('Number of Postings vs Average Salary (Last 2 Years)')
ax2.set_xlabel('Number of Postings')
ax2.set_ylabel('Average Salary ($)')
ax2.grid(True)

# 3. Top 10 Job Categories
category_counts_pd.plot(kind='bar', x='Job Category', y='count', ax=ax3)
ax3.set_title('Top 10 Job Categories by Number of Postings')
ax3.set_xlabel('Job Category')
ax3.set_ylabel('Number of Postings')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()""")
]

nb.cells = cells

# Write the notebook to a file
with open('jupyter/notebook/nyc_jobs_analysis.ipynb', 'w') as f:
    nbf.write(nb, f) 