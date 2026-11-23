# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 06:01:23 2026
Name: Industrial IoT Telemetry Pipeline - Annotated Build Walkthrough
Purpose: Documents the iterative prototyping, engine-level reasoning (Pandas eager
vs. Polars lazy), and edge-case testing behind `Industrial_IoT_Telemetry_Pipeline.py`.
Scenario: A regional smart-grid sensor network emits messy, unstructured log arrays. The objective is to clean the telemetry strings, parse geographical spatial coordinates embedded inside text flags, classify risk levels without using heavy Python loops, and calculate isolated ranking metrics per zone.
"""

import csv
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# We begin with our given, raw data:
sensor_telemetry_raw = {
    " SENSOR_ID  ": ["  sn-101 ", " sn-102", "sn-103 ", "SN-101", "sn-104", "sn-102 ", "  sn-105 "],
    "raw_logs": [
        "ERROR|LOC:40.7128,-74.0060|VOLT:240.5",
        "WARN|LOC:39.9526,-75.1652|VOLT:missing",
        "OK|LOC:34.0522,-118.2437|VOLT:120.2",
        "ERROR|LOC:40.7128,-74.0060|VOLT:285.0",
        "OK|LOC:41.8781,-87.6298|VOLT:118.9",
        "ERROR|LOC:39.9526,-75.1652|VOLT:310.1",
        "WARN|LOC:null|VOLT:190.4"
    ],
    "timestamp": ["2026-08-01 10:00", "2026-08-01 10:05", "2026-08-01 10:10", "2026-08-01 10:15", "2026-08-01 10:20", "2026-08-01 10:25", "2026-08-01 10:30"]
}


# =============================================================================
# %%PART 0: Converting our Raw Dictionary Data into a Workable Form
# =============================================================================
# We are creating a  DataFrame object using Pandas and a LazyFrame object using Polars. However, we first need to convert our `dict` type sensor_telemetry_raw object into a csv object. To do this, we will create a new physical csv file for Pandas and Polars to read. 
# First, we will need to set the path for this file to be in the directory of this working file.
script_dir = Path.cwd()
csv_file_path = script_dir / "sensor_telemetry_raw.csv"

# Next, we will open this filepath to write as a new_file and write the sensor_telemetry_raw key values to it as column headers. We make use of `with open()` to open the file, keep it open while we write to it, and then close it when we are done. We use "w" to indicate that this operation is in "write-only mode".
with open(csv_file_path, "w", newline="") as new_file:
    writer = csv.writer(new_file)
    writer.writerow(list(sensor_telemetry_raw.keys()))

# For our next step, we need to make use of the .values() method of sensor_telemetry_raw. First, we must coerce them into a list. Once we have that list version of them, we take a moment to consider the data. What do we have in this list of data? Letting our eyes adjust, we notice that we have is a list of lists: first, a list of 3 sublists, each of which contains exactly 7 elements. We note that each sublist of values corresponds to a single dictionary key in the original dataset. 
dt_val = list(sensor_telemetry_raw.values())

# We are definitely getting closer. We realize that we can't just write this object onto our virtual csv as it stands: it doesn't match our current written values, which consists of a list of 3 elements, each being a dictionary key value from the sensor_telemetry_raw dictionary. This leads us to realize two things: (1), that the form that each additional row addended to string_buffer via the writer must be in the form of lists of exactly 3 values, and (2) that each value of the three sublists of dt_val correspond to the key entry of raw dictionary keys that was written onto string_buffer as its first row. We can now summarize what we need and what we have:
    # What we need: we need to add the values of sensor_telemetry_raw into string_buffer from a list of sublists; each sublist needs to contain three values, and the value in each index must correspond to its initial dictionary key value in the index of the first entry.
    # What we have: we have a list of 3 sublists each containing 7 values, dt_val. The values of each sublist correspond to the key values of their index matched to the index of the initial key value in the first row.
# If we think about what we have in terms of matrices (and allow for a small abuse of notation), then dt_val is analogous to a matrix with 3 rows and 7 columns, with each sublist corresponding to a row of this matrix and the entry for each element of each sublist corresponding to the column of this matrix. Similarly, we can think of string_buffer being the header of a matrix of data consisting of 1 row with 3 values. In this formulation, we see that dt_val is very close to what we are wanting to find; if we could just find what is analogous of this matrix that is analogous to dt_val, then we would have what we need. Thankfully, we can get exactly this: by utilizing the `zip()` function and applying it to the unpacked dt_val, our output will be exactly analogous to a 7x3 matrix where the values of each column correspond to their initial dictionary key value in the header column; this is analogous to a list of 7 sublists each containing 3 values where the value in each index corresponds to its initial dictionary key value in the index of the header entry. This is precisely what we were looking for.
tr_dt_val = list(zip(*dt_val))

# Now that we have the rest of our information for our .csv file, we can write it to our csv_file_path. This time, we will use "a" to indicate that this operation is done in "append mode".
with open(csv_file_path, "a", newline = "") as new_file:
    writer = csv.writer(new_file)
    writer.writerows(tr_dt_val)

# We can now use pd.read_csv and pl.scan_csv to read our new csv into a Pandas DataFrame and Polars LazyFrame, respectively

df_pd_raw = pd.read_csv(csv_file_path)
lf_pl_raw = pl.scan_csv(csv_file_path)

# =============================================================================
# END PART 0
# =============================================================================
# %%===========================================================================



#### CORE EXECUTION PIPELINES ####
# Perform these transformations using a strict single method chain for Pandas, and a strict single expression chain for Polars. No intermediate variables allowed!


# =============================================================================
# %%PART 1: Pandas Implementation with Pandas DataFrame Objects
# =============================================================================

# %%% Task 1: Ingestion, Header Normalization, and Token Extraction
# Standardize column headers to lowercase and strip all structural whitespace.
# Clean up sensor_id strings to establish a uniform uppercase format ("SN-101") with no 
#   trailing padding.
# Parse the raw_logs string column using Regular Expressions to extract three clean features:
    # (1) status: The alphabetic string preceding the first pipe (|) delimiter.
    # (2) latitude: The numeric coordinate text isolated immediately following the LOC:
    #       flag.
    # (3) voltage: The floating-point decimal trailing the VOLT: flag, coercing non-numeric 
    #       text like "missing" into true null values.
    
# -----(A)----- 
# We have a few subtasks ahead of us to complete Task 1. First, let us focus on the first part of Task 1, which we will call (A): standardizing headers to be all lowercase and without any structural whitespace. These operations are fairly basic.
test1 = df_pd_raw.rename(columns=lambda c: c.strip().lower())


# -----(B)----- 
# Our next subtask is very similar to (A), just now being applied to row items instead of column headers.  Before we begin however, let's make sure that sensor_id and raw_logs are of str type. We will use the .astype() method for coercion, and we will call a dictionary to it with the keys as sensor_id and raw_logs
test1 = test1.astype({"sensor_id": str, "raw_logs": str})

# With that out of the way, we are going to remove the whitespace from each entry of "sensor_id" as well as coerce each entry to lowercase. Because df["sensor"] is a Pandas Series object, we cannot call the strip() and lower() string methods directly to it; however, the using the `.str` accessor of a Pandas Series vectorizes whatever string function is passed directly after it to apply to the entire Pandas Series that precedes it. Chaining `str.strip()` to `str.lower()` simply requires inserting a `.` in between them.
test1 = test1.assign(sensor_id = lambda df: df["sensor_id"].str.strip().str.upper())


# -----(C)----- 
# Our final subtask for Task 1 is to parse the raw_logs string column to extract three particular attributes from each entry. Our next task is the tough one: we need to parse raw_logs into three new attributes: status, latitude, and voltage. Let us start by developing our regular expression needed to parse a string like this. We will define `tst_str` to be the raw_logs expression from the first row of data.
tst_str = test1.loc[0, "raw_logs"]
    
# What is it that we want our regular expression to do? We want to extract all of the values prior to the first | symbol (This is our new attribute, "status"); we want to extract everything between the first colon and the comma (This is our new attribute, "latitude"); and we want to extract everything after "VOLT:" (The second variable that splits off from this last split aka "the leftover" is our last new attribute, "voltage"). We will need to be mindful of the data types for each new attribute: string, numerical, and numerical, respectively. Lastly, for the second and third attributes, we will need to coerce missing values in some manner.
# First, we create our regular expression. 
    # The first regex, reg_status, reads as "pattern: group starts at the very beginning of the string, and contains at least one instance of an alphabetic character (but possibly more, with no upper bound); group is bounded by the occurrence of the | character.
reg_status = r"(\A(?:[a-zA-Z])+)\|"
    # The second regex, reg_latitude, reads as "pattern: a colon bounds the group to the left; the group then consists of either one single chain of {at least one decimal digit (but possibly more w/ no upper bound) followed by a single period followed by at least one more decimal digit (but possibly more w/ no upper bound)} or any number (including none) of alphabetic characters."
reg_latitude = r":((?:(?:\d+\.\d+))|(?:[a-zA-Z]*))"
    # The third regex, reg_voltage, reads as "pattern: group is bordered to the left by "VOLT:"; the group then consists of possibly a single (or no) chain of {at least one decimal digit (but possibly more w/ no upper bound) followed by a single period followed by at least one more decimal digit (but possibly more w/ no upper bound)} or any number (including none) of alphabetic characters. The group is then bounded on the right by the end of the string."
reg_voltage = r"VOLT:((?:\d+\.\d+)?|(?:[a-zA-Z]*))\Z"

# Calling these into re.search() along with tst_str yields <=> `re.search(reg#, tst_str)[1]` gives us 'ERROR', '40.7128', and '240.5': exactly what we want! We now want to take these regular expressions and apply each of them to the raw_logs column for each regular expression and assign them to the new headers "status", "latitude", and "voltage", respectively.
test1 = test1.assign(status = lambda df: df["raw_logs"].str.extract(reg_status))
test1 = test1.assign(latitude = lambda df: df["raw_logs"].str.extract(reg_latitude)).assign(latitude = lambda df: pd.to_numeric(df["latitude"], errors="coerce"))
test1 = test1.assign(voltage = lambda df: df["raw_logs"].str.extract(reg_voltage)).assign(voltage = lambda df: pd.to_numeric(df["voltage"], errors="coerce"))

# Let's check to make sure all of our new values match up with raw_logs. We will make a new table, print the table, and also print the tables' dtypes.
pd_task1 = test1[["raw_logs","status","latitude","voltage"]]
print(f"Task 1 relevant Pandas DataFrame:\n {pd_task1}")
print(f"Task 1 relevant column datatypes:\n {pd_task1.dtypes}\n")


# To finish Task 1, we will pull everything together into a single Pandas chain. This task focused on cleaning the raw data, so this name fits.
df_pd_cleaned = (
    df_pd_raw
    .rename(
        columns=lambda c: c.strip().lower()
        )
    .astype(
        {"sensor_id": str, "raw_logs": str}
        )
    .assign(
        sensor_id = lambda df: df["sensor_id"].str.strip().str.upper()
        )
    .assign(
        status = lambda df: df["raw_logs"].str.extract(r"(\A(?:[a-zA-Z])+)\|")
        )
    .assign(
        latitude = lambda df: df["raw_logs"].str.extract(r":((?:(?:\d+\.\d+)+)|(?:[a-zA-Z]*))")
        )
    .assign(
        latitude = lambda df: pd.to_numeric(df["latitude"], errors="coerce")
            )
    .assign(
        voltage = lambda df: df["raw_logs"].str.extract(r"VOLT:((?:\d+\.\d+)?|(?:[a-zA-Z]*))\Z")
        )
    .assign(
        voltage = lambda df: pd.to_numeric(df["voltage"], errors="coerce")
        )
    )


# %%% Task 2: Priority Conditionals and Risk Profile Classification
# Derive a new categorical attribute named risk_profile based on the operational thresholds:
    # If status is "ERROR" and voltage exceeds 250.0 --> label it "CRITICAL".
    # If status is "ERROR" or "WARN" --> label it "HIGH".
    # Otherwise --> label it "STABLE".
    # Engine Constraints: Pandas must execute this vectorization step within .assign() via
    #   np.select().
    

# We can use np.select() to evaluate all of these at once. We define the two stated conditions in (A) and (B) as cond_A and cond_B, respectively. If a condition choice satisfies both conditions, then the leftmost condition that evaluates as True will take precedence. Since something that satisfies cond_A will always satisfy cond_B, cond_A will be listed first. Finally, if both conditions evaluate to False, the value of parameter 'default' will be assigned.
test2 = df_pd_cleaned.assign(risk_profile = lambda df: np.select([(df["status"] == 'ERROR') & (df["voltage"] > 250), df["status"].str.fullmatch(r"ERROR|WARN")], ["CRITICAL", "HIGH"], "STABLE"))
pd_task2 = test2[["status","voltage","risk_profile"]]
print(f"Task 2 relevant Pandas DataFrame:\n {pd_task2}\n")


# We finish Task 2 with a single Pandas chain. Since we just finished a "classification" task, the following new name seems appropriate.
df_pd_classified = (
    df_pd_cleaned
    .assign(
        risk_profile = lambda df: np.select(
            condlist=[
                (df["status"] == 'ERROR') & (df["voltage"] > 250),
                df["status"].str.fullmatch(r"ERROR|WARN")
                ], 
            choicelist=["CRITICAL", "HIGH"], 
            default = "STABLE"
            )
        )
    )


# %%% Task 3: Spatial Filtering and Anomaly Data Cleansing
# Filter out and drop all rows containing missing values or null entries inside the extracted
#   latitude and voltage columns.

# We can accomplish this with a single query. Our process is this: for .query, we must wrap everything in quotations because it is using a different engine (numexpr); next, we cannot try to invoke outside methods like Pandas pd or numpy np within the query; further, since the numexpr uses natural language, we use "and" instead of "&"; penultimately, we can logically negate a boolean list (or Pandas Series) using "not"; finally, we can use the method `isna()` on a Pandas Series to check for values that are NA, which includes the numpy.NaN value that we have in our dataframes. All together, we go for the expression combination: "we will query for voltage column elements that do NOT evaluate to True for .isna() as well as the latitude column elements that do NOT evaluate to True for .isna(). In numexpr form, "not voltage.isna() and not latitude.isna()":
test3 = df_pd_classified.query("not voltage.isna() and not latitude.isna()")
pd_task3 = test3[["latitude","voltage"]]
print(f"Task 3 relevant Pandas DataFrame:\n {pd_task3}\n")

# We can finally write our final Task 3 single Pandas chain.
df_pd_defined = (
    df_pd_classified
    .query("not voltage.isna() and not latitude.isna()")
    )


# %%% Task 4: Partitioned Analytics and Windowed Voltage Ranking
# Compute a relative ranking metric across the dataset while fully preserving the row density of
#   individual log records (no grouping or structural row reduction):
    # Create a column named voltage_rank_in_sensor that ranks the row's voltage relative to that
    #   specific sensor_id, ordered from highest to lowest voltage.
    # Engine Constraints: Pandas must leverage .groupby().rank().
    # Output a Pandas DataFrame object that displays the number of each unique sensor_id values

# By the column's name: "voltage_rank_in_sensor", we see that we will rank the row's voltage relative to the sensor_id. First, we will groupby sensor_id. We use the .unique() method on the sensor_ID Pandas Series to find that there are 4 unique values in the remaining dataset. The same procedure shows all voltage values are unique.
test4 = df_pd_defined
print(test4["sensor_id"].unique(),"| values = ", test4["sensor_id"].unique().size)
print(test4["voltage"].unique(),"| values = ",test4["voltage"].unique().size)

# Here is our procedure and an explanation of what it is we are doing. First, we want to perform a groupby "sensor_id". What this means is that, when we calculate the ranks for each index value later, we will have 4 competing groups: 'SN-101' --> Index 0 & 3, 'SN-103' --> Index 2, 'SN-104' --> Index 4, and 'SN-102' --> Index 5. Because .rank() is telling us who gets in first, second, third, etc for each group, we see that we will have 3 groups with everyone rank 1 and 1 group with a rank 1 and a rank 2.
# Once we have our .groupby() applied to sensor_id, we select out of our dataframe the one column we are interested in ranking by: the voltage column. We slice this out with the colon notation. Next, we apply the .rank() method to our chain. We accept the default methods, but because we want to rank voltage from highest voltage to lowest voltage, we set the parameter ascending to equal False.
test4.groupby(["sensor_id"])["voltage"].rank(ascending=False)
# Finally, we will perform an assignment of the result of this ranking to our dataframe. We use .assign(), name the new column "voltage_rank_in_sensor", and define our dataframe to equal this assignment.
test4 = test4.assign(voltage_rank_in_sensor = lambda df: df.groupby(["sensor_id"])["voltage"].rank(ascending=False))
pd_task4 = test4[["sensor_id","voltage","voltage_rank_in_sensor"]]
print(f"Task 4 relevant Pandas DataFrame:\n {pd_task4}\n")

# We construct our final Task 4 single Pandas chain.
df_pd_grouprank = (
    df_pd_defined
    .assign(
        voltage_rank_in_sensor = lambda df: df
        .groupby(["sensor_id"])["voltage"]
        .rank(ascending=False)
        )
    )




# =============================================================================
# END PART 1
# =============================================================================
# %%===========================================================================



# =============================================================================
# %%PART 2: Polars Implementation with Polars LazyFrame Objects
# =============================================================================

# %%% Task 1: Ingestion, Header Normalization, and Token Extraction
# Standardize column headers to lowercase and strip all structural whitespace.
# Clean up sensor_id strings to establish a uniform uppercase format ("SN-101") with no 
#   trailing padding.
# Parse the raw_logs string column using Regular Expressions to extract three clean features:
    # (1) status: The alphabetic string preceding the first pipe (|) delimiter.
    # (2) latitude: The numeric coordinate text isolated immediately following the LOC:
    #       flag.
    # (3) voltage: The floating-point decimal trailing the VOLT: flag, coercing non-numeric 
    #       text like "missing" into true null values.
    
# -----(A)----- 
# For subtask A, we simply need to coerce the headers of our lazyframe to lowercase and eliminate whitespace characters from them. We will accomplish this with the pl lazyframe .rename() method calling a dictionary comphrehension. The syntax for this is `{'OLD_NAME': 'NEW_NAME'}, and we achieve this in the dict comprehension by having 'c: c.strip().lower()' where c is the selection of an interative for loop over the list of the names from the schema of lf_pl_raw.
pl_tst1 = lf_pl_raw.rename({c: c.strip().lower() for c in lf_pl_raw.collect_schema().names()})

# -----(B)----- 
# For subtask 2, we normalize the text values within the "sensor_id" column. We accomplish this by invoking the LazyFrame `.with_columns()` method and passing a chain of expressions applied to the `pl.col("sensor_id")` symbolic selector.
# Inside the column expression, we access the `.str` accessor property to enter the string namespace, and call the `.to_uppercase()` method. Because Polars string methods return a base Expression object, we chain a second operation by accessing the `.str` accessor again and calling the `.strip_chars()` method to clear extraneous whitespace. Finally, we call `.collect()` to execute the query plan and evaluate our output DataFrame..
pl_tst1.with_columns(pl.col("sensor_id").str.to_uppercase().str.strip_chars()).collect()


# -----(C)----- 

# We will test our expressions against a standard case, a special latitude case, and a special voltage case. We will hence define three variables with sequence unpacking to have a test case for each variable. 
check1, check3, check2 = pl_tst1.select("raw_logs").collect().to_series()[0,1,6]
    # We first construct the regex pattern for status. What we desire is to obtain an expression of a sequence of alphabetic characters (of either case) that can be arbitrarily long. These expressions have the added condition that they must be preceded by the start of the string and succeeded by the '|' character.
regex_status = r"^([a-zA-Z]+)\|"

    # We next work on the regex pattern for latitude. What we desire is to obtain an expression of a sequence of decimal digits of any (nonzero) length followed by a single period ("decimal point") followed by any (nonzero) number of more decimal digits, with an optional minus ("negative sign") in front with the look-ahead condition requiring the presence of a comma, or alternatively, an expression of a sequence of alphabetic characters (of either case) that can be arbitrarily long with the look-ahead condition requiring the presence of a | symbol. These expressions have the added condition that they must be preceded by "LOC:" in the string.
        # NOTE: The optional negative sign and look-ahead are NOT required for this code to work, but they do give the regex pattern more general power
regex_lat_looks = r"LOC:((?:-?\d+\.\d+(?=,))|(?:[a-zA-Z]+(?=\|)))"
# ...HOWEVER, Polars does not support lookahead or lookbehind regex! So, we try to get this general power without them. We come up with an amazing use of a negated character class: using the construction `[^~]`, we are setting up a statement that says "match exactly one character, so long as it is NOT any input that follows. So, for our case, we want to anchor ourselves on the left with "LOC:", as before, and what we want is to extract whatever comes next before we either hit the `comma` after the latitude that separates it from the longitude or before the `pipe` that separates the error/missing value from "VOLT" that comes after it. Our negated character class is hence `[^,|]`, which translates to "match any single character, provided it is neither a comma nor a pipe." To get all of the characters we desire, we will use `+` after it to indicate "at least one occurence up to a boundless number of occurrences. Combining this with our anchor, we end up building the statement "Start after `LOC:` and begin to capture each successive character that is neither a comma nor a pipe. Once you encounter a comma or a pipe, so long as you've captured a single character, the `+` quantifier is satisfied. Thus, return the group of characters captured." This is represented in regex below.
        # NOTE: So why didn't the pipe need to be escaped this time? In the regex lookahead, we had
        # to make use of the backslash `\` in order to properly match string characters. Generally
        # (outside of these brackets), the pipe symbol `|` means "Alternation" (Logical OR) and is 
        # a *structural command character*. This means that if you want it to match a literal line
        # separator within text, you must escape it using the backslash `\|`. However, when within
        # brackets, alternation doesn't exist: a character class is *already* an implied list of
        # literal choices. Thus, within the brackets, the parsing engine treats the pipe character
        # `|` as a literal pipe character.
regex_lat_neg = r"LOC:([^,|]+)"


    # Our final regex pattern is for voltage. What we desire is to obtain an expression of a sequence of decimal digits of any (nonzero) length followed by a single period ("decimal point") followed by any (nonzero) number of more decimal digits, or alternatively, an expression of a sequence of alphabetic characters (of either case) that can be arbitrarily long. These expressions have the added condition that they must be preceded by "VOLT:" and succeeded by the end of the string.
regex_volt = r"VOLT:((?:\d+\.\d+)|[a-zA-Z]+)$"

# We now test how a chain of subtask (C) would appear. We also would like to coerce latitude and voltage values to numeric. We will also coerce the errors.
pl_tst1.with_columns(status = pl.col("raw_logs").str.extract(regex_status), latitude = pl.col("raw_logs").str.extract(regex_lat_neg).str.to_decimal(scale=4), voltage = pl.col("raw_logs").str.extract(regex_volt).str.to_decimal(scale=1)).collect()


# We now complete Task 1 by performing all of the above in a single chain, naming it to a new Polars LazyFrame variable in order to use in the next task. We will finish by outputting the final Polars DataFrame with .collect(). Additionally, we will assign the relevant dataframe information to a new variable and print it.

lf_pl_cleaned = (
    lf_pl_raw
    .rename({c: c.strip().lower() for c in lf_pl_raw.collect_schema().names()})
    .with_columns(
        pl.col("sensor_id").str.to_uppercase().str.strip_chars(),
        status = pl.col("raw_logs").str.extract(r"^([a-zA-Z]+)\|"),
        latitude = pl.col("raw_logs").str.extract(r"LOC:([^,|]+)").str.to_decimal(scale=4), 
        voltage = pl.col("raw_logs").str.extract(r"VOLT:((?:\d+\.\d+)|[a-zA-Z]+)$")
        .str.to_decimal(scale=1)
        )
    )

pl_task1 = lf_pl_cleaned.collect()
pl_task1 = pl_task1.select("sensor_id","raw_logs","status","latitude","voltage")
print(f"Task 1 relevant Polars DataFrame:\n {pl_task1}")
print(f"Task 1 relevant column datatypes:\n {pl_task1.dtypes}\n")



# %%% Task 2: Priority Conditionals and Risk Profile Classification
# Derive a new categorical attribute named risk_profile based on the operational thresholds:
    # If status is "ERROR" and voltage exceeds 250.0 --> label it "CRITICAL".
    # If status is "ERROR" or "WARN" --> label it "HIGH".
    # Otherwise --> label it "STABLE".
    # Engine Constraints: Polars must utilize native pl.when().then().otherwise() syntax.
    
# We will first construct the conditional for the statements above:
    # (A) Status = ERROR AND voltage > 250. As statements inside of pl, this translates to `pl.col("status") == "ERROR"  & pl.col("voltage") > 250`
    # (B) Status = ERROR or WARN. As statements inside of pl, this translates to `pl.col("status") == "ERROR" | "WARN"`
    # (C) Since we are using pl.when().then().otherwise(), this condition just defaults to .otherwise().
    # We note that, although condition (A) and condition (B) could BOTH occur at the same time, when-then-otherwise operates on a left-to-right basis, so writing this in order will resolve all issues.
# We are now ready to execute the code as a single Polars chain. Due to the way Polars runs internal processes ("implicit literal coercion"), we need to indicate when a string is actually a literal string as opposed to a column header in .then() and .otherwise() methods. For this, we use pl.lit().
lf_pl_classified = (
    lf_pl_cleaned.with_columns(
        risk_profile = pl.when((pl.col("status") == "ERROR") & (pl.col("voltage") > 250))
        .then(pl.lit("CRITICAL"))
        .when(pl.col("status").is_in(["ERROR", "WARN"]))
        .then(pl.lit("HIGH"))
        .otherwise(pl.lit("STABLE"))
        )
    )

pl_task2 = lf_pl_classified.collect()
pl_task2 = pl_task2.select("sensor_id","status","voltage","risk_profile")
print(f"Task 2 relevant Polars DataFrame:\n {pl_task2}")
print(f"Task 2 relevant column datatypes:\n {pl_task2.dtypes}\n")



# %%% Task 3: Spatial Filtering and Anomaly Data Cleansing
# Filter out and drop all rows containing missing values or null entries inside the extracted
#   latitude and voltage columns.

lf_pl_defined = (
    lf_pl_classified.drop_nulls()
    )

pl_task3 = lf_pl_defined.collect()
pl_task3 = pl_task3.select("sensor_id","latitude","voltage")
print(f"Task 3 relevant Polars DataFrame:\n {pl_task3}")
print(f"Task 3 relevant column datatypes:\n {pl_task3.dtypes}\n")


# %%% Task 4: Partitioned Analytics and Windowed Voltage Ranking
# Compute a relative ranking metric across the dataset while fully preserving the row density of
#   individual log records (no grouping or structural row reduction):
    # Create a column named voltage_rank_in_sensor that ranks the row's voltage relative to that
    #   specific sensor_id, ordered from highest to lowest voltage.
    # Engine Constraints: Polars must use an expression block evaluated over the partition via
    #   .over().
    # Output a Polars DataFrame object that displays the number of each unique sensor_id values.
    #   NOT required to be part of the single expression Polars query chain
    
# Our procedure is fairly straightforward: first, we will use .with_columns(voltage_rank_in_sensor = ...) to create a new column named "voltage_rank_in_sensor". We set this equal to the following chain from left to right: 
    # `pl.col("voltage")`: We take the voltage column...
    # `.rank("ordinal", descending=True)`: ...calculate its ordinal rank from highest to lowest...
    # `.over("sensor_id")`: ...and perform that entire calculation over independent "sensor_ID" groups

lf_pl_grouprank = (
    lf_pl_defined.with_columns(
        voltage_rank_in_sensor = pl.col("voltage").rank("ordinal", descending=True)
        .over("sensor_id")
        )
    )

pl_task4 = lf_pl_grouprank.collect()
pl_task4 = pl_task4.select("sensor_id","voltage","voltage_rank_in_sensor")
print(f"Task 4 relevant Polars DataFrame:\n {pl_task4}")
print(f"Task 4 relevant column datatypes:\n {pl_task4.dtypes}\n")



# %%% Task 5: Query Plan Inspection and Lazy Engine Performance Audit
# Evaluate the compiled query graph of the finalized Polars pipeline:
    # Output the optimized logical execution tree to the terminal console using the .explain()
    #   inspector to audit predicate pushdowns executed by the Polars optimization tree.
    # Output the full, final Polars DataFrame.

# Since we completed the entirety of Part 2 using a Polars LazyFrame the whole way, all we need to do is to invoke the `.explain()` method on our final Polars LazyFrame, lf_pl_grouprank. Before we do this, for fun, let's go ahead and convert our timestamp column to a datetime type.
lf_pl_final =(
    lf_pl_grouprank.with_columns(pl.col("timestamp").str.to_datetime())
              )

# We now print the final Polars LazyFrame optimization and the final Polars DataFrame that results.
part2_explanation = lf_pl_final.explain()
print("\n-------------------------------------------------------------------------------\n")
print(f"Part 2: Polars LazyFrame Optimization Explanation:\n {part2_explanation}")
part2_polars_dataframe = lf_pl_final.collect()
print("\n-------------------------------------------------------------------------------\n")
print(f"Part 2: Polars DataFrame:\n {part2_polars_dataframe}\n")

    
    
    
    
    
    
    
    