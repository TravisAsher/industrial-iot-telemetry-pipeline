# Industrial IoT Telemetry Pipeline: A Dual-Engine Architecture Study

An end-to-end data engineering case study demonstrating high-performance text parsing, schema normalization, and windowed analytics executed in parallel across both **Pandas (Eager)** and **Polars (Lazy)** processing engines.

## System Architecture and Core Engineering Phases
This pipeline ingests raw, malformed industrial sensor logs containing inconsistent tracking keys, unstructured string attributes, and missing data anomalies. The workflow is split into five distinct production phases executed under strict single-chain formatting constraints:

  * **Task 1**: String Isolation and Regular Expressions
    Standardizes column headers to lowercase and strips structural whitespace. Cleans up sensor IDs
    to establish a uniform uppercase format ("SN-101"). Parses raw string logs via customized
    negated character classes (`[^,|]+` in Polars) to isolate status tokens, high-precision
    latitudinal coordinates, and numerical voltages while gracefully bypassing lookahead limits.


  * **Task 2**: Vectorized Conditional Logic
    Vectorizes multi-condition operational thresholds without iterative loops. Evaluates risk
    hierarchies left-to-right using explicit condition priority queues via `np.select()` in Pandas
    and native `pl.when().then().otherwise()` syntax blocks with implicit literal coercions
    (`pl.lit()`) inside the Polars lazy query plan.


  * **Task 3**: Spatial Filtering and Anomaly Data Cleansing
    Sanitizes missing or text-corrupted geographical attributes using high-efficiency row-filtering
    operations. Leverages natural language strings inside the `numexpr` engine for Pandas (`.query`)
    and utilizes deferred null transformations via `.drop_nulls()` to purge anomalies in Polars.


  * **Task 4**: Advanced Window Aggregations
    Computes analytical, row-level ordinal voltage ranks partitioned over independent sensor groups
    without collapsing structural log density or reducing rows. Leverages native window evaluation
    expressions via `.over("sensor_id")` in Polars and `.groupby().rank(ascending=False)` inside
    the Pandas eager framework.


  * **Task 5**: Lazy Engine Performance Check
    Evaluates compiled query graphs using the `.explain()` inspector to analyze optimization
    mechanics. This stage profiles how the Polars optimizer rearranges the processing lifecycle,
    hoisting filters and pushing down projection rules directly into the underlying file scan.


## Repository Structure

The tracking repository is structured across dedicated functional layers:

* `Industrial_IoT_Telemetry_Pipeline.py` - Production pipeline featuring optimized, single-method
  execution chains utilizing **Pandas** DataFrame and **Polars** LazyFrame
* `Project_Specification.Rmd`: Cleaned, synchronized project specification briefing written in RMarkdown.
* `Project_Specification.pdf` - Rendered requirements brief optimized for evaluation and presentation
* `matrix_transformations_and_regex_study.py` - Details my investigations in various topics over the course of this project, including the `zip()` function, regular expressions, and more <br>
  > <sub> NOTE: This is a work in progress
* `.gitignore` - Explicitly handles global Python cache layers and blocks raw CSV data dumps.


## Key Technical Solutions Included

* **Matrix Transpositions**: Implemented high-performance dictionary value vertical pivoting by
  applying unpacking operators directly onto the native Python zip utility via `zip(*dt_val)`.
* **Regular Expressions**: Purpose of this document was to understand regular expressions further.
  Work in progress
  
#### THE FOLLOWING ARE ALL WORKS IN PROGRESS
* **Shape Consistency**: Resolved Pandas extraction `TypeErrors` by enforcing `expand=False` inside
  the `.str.extract()` sequence to preserve 1-D Series dimensions during token parsing.
* **Fixed-Point Precision**: Maintained spatial geographical coordinate accuracy by utilizing
  Polars' precise fixed-point Decimal engine (`scale=4` and `scale=1`) to eliminate float flaws.
* **Typing Synchronization**: Coerced temporal logging values into matching native datetimes
  across both engines to guarantee end-to-end structural schema symmetry.




## Query Plan Evaluation and Optimizer Deep Dive

To evaluate the structural execution differences between Python-eager meta-mapping and purely
deferred lazy evaluations, back-to-back engine compilations were audited via Polars `.explain()`.

The benchmark compared an upfront dictionary comprehension schema lookup against a deferred
functional lambda execution pattern:

### 1. Dictionary Comprehension (Eager Meta-Mapping)
```python
lf_pl_raw.rename({c: c.strip().lower() for c in lf_pl_raw.collect_schema().names()})
```

### 2. Functional Lambda (Pure Lazy Deferred)
```python
lf_pl_raw.rename(lambda c: c.strip().lower())
```

### Compiler Compilation Analysis

The Polars Query Planner generated **100% identical physical execution graphs** for both
implementations, demonstrating highly resilient internal query graph assembly:

```text
  WITH_COLUMNS:
 [col("voltage").rank().over([col("sensor_id")]).alias("voltage_rank_in_sensor")] 
  FILTER col("risk_profile").is_not_null()
  FROM
     WITH_COLUMNS:
     [when([([(col("status")) == ("ERROR")]) & ([(col("voltage")) > (250.0)])]).then("CRITICAL").otherwise(when(col("status").is_in([["ERROR", "WARN"]])).then("HIGH").otherwise("STABLE")).alias("risk_profile")] 
      FILTER [([([([([(col("raw_logs").is_not_null()) & (col("voltage").is_not_null())]) & (col("timestamp").is_not_null())]) & (col("latitude").is_not_null())]) & (col("status").is_not_null())]) & (col("sensor_id").is_not_null())]
      FROM
         WITH_COLUMNS:
         [col("sensor_id").str.to_uppercase().str.strip_chars([null]), col("timestamp").str.strptime(["raise"]), col("raw_logs").str.extract(["^([a-zA-Z]+)\|"]).alias("status"), col("raw_logs").str.extract(["LOC:([^,|]+)"]).str.to_decimal().alias("latitude"), col("raw_logs").str.extract(["VOLT:((?:\d+\.\d+)|[a-zA-Z]+)\$"]).str.to_decimal().alias("voltage")] 
          SELECT [col(" SENSOR_ID  ").alias("sensor_id"), col("raw_logs"), col("timestamp")]
            Csv SCAN [sensor_telemetry_raw.csv]
            PROJECT */3 COLUMNS
```


### Optimization Mechanics Dissected

* **Projection Pushdown Integration:** Polars completely eliminated the overhead of an independent
  `RENAME` operation. The engine optimized the operation away by injecting the name-cleaning rules
  directly into the initial `SELECT` statement mapping inside the `Csv SCAN` projection boundary.
* **Predicate Pushdown Alignment:** The mid-pipeline `.drop_nulls()` sanitation step was
  automatically hoisted down the execution stack by the optimizer. It compiled into a unified,
  multi-column `FILTER` node positioned immediately after token extraction, ensuring rows
  containing anomalies are eliminated *before* computing resource-heavy conditional structures.
* **Architectural Tradeoffs:** While local execution is identical due to low-latency disk
  indexing, the **Lambda/Selector approach** is structurally preferred in cloud-distributed
  environments (e.g., S3/GCS data lakes) because it keeps the optimization tree completely
  independent of remote network metadata requests during the initial graph assembly stage.

#### NOTE: THIS IS ALL STILL A WORK IN PROGRESS. MORE COMING SOON.
