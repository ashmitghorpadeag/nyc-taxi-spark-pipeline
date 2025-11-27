from pyspark.sql.functions import *
df = spark.read.parquet("/Workspace/Users/ashmit.ghorpade2@mail.dcu.ie/yellow_tripdata_2024-01.parquet")
print(f"Total rows: {df.count()}")
df.printSchema()
df.show(5)
before = df.count()
df_clean = df.filter(
    (col("passenger_count") > 0) &
    (col("trip_distance") > 0) &
    (col("trip_distance") < 100) &
    (col("total_amount") >= 2.50)
)
after = df_clean.count()
removed = before - after
print(f"Rows before cleaning: {before}")
print(f"Rows after cleaning: {after}")
print(f"Removed {removed} bad records ({removed/before*100:.2f}%)")
df_enriched = df_clean.withColumn(
    "pickup_time", to_timestamp(col("tpep_pickup_datetime"))
).withColumn(
    "dropoff_time", to_timestamp(col("tpep_dropoff_datetime"))
).withColumn(
    "trip_duration_min", 
    (unix_timestamp(col("dropoff_time")) - unix_timestamp(col("pickup_time"))) / 60
).withColumn(
    "hour_of_day", 
    hour(col("pickup_time"))
).withColumn(
    "day_of_week",
    dayofweek(col("pickup_time"))
).withColumn(
    "fare_per_mile",
    col("total_amount") / col("trip_distance")
)

print("New features added:")
df_enriched.select("hour_of_day", "day_of_week", "trip_duration_min", "fare_per_mile", "total_amount").show(20)
df_summary = df_enriched.groupBy("hour_of_day", "day_of_week").agg(
    avg("trip_distance").alias("avg_distance"),
    avg("total_amount").alias("avg_fare"),
    avg("trip_duration_min").alias("avg_duration"),
    count("*").alias("trip_count")
).orderBy("hour_of_day", "day_of_week")

print("AGGREGATED RESULTS (by hour and day):")
df_summary.show(100)

print(f"\nTotal combinations: {df_summary.count()}")
from pyspark.sql.window import Window

print("\n" + "="*70)
print("EXTENDED ANALYSIS: ANOMALIES, SEGMENTATION & PATTERNS")
print("="*70)
print("\n[ANALYSIS 1] Detecting anomalous trips...")
window_spec = Window.partitionBy("hour_of_day").orderBy("total_amount")
df_with_percentile = df_enriched.withColumn(
    "fare_percentile", percent_rank().over(window_spec)
).withColumn(
    "is_anomaly", when((col("fare_percentile") < 0.05) | (col("fare_percentile") > 0.95), 1).otherwise(0)
)

anomalies = df_with_percentile.filter(col("is_anomaly") == 1)
print(f"Anomalies found: {anomalies.count()} trips (unusual fares)")
anomalies.select("hour_of_day", "total_amount", "trip_distance").show(10)
print("\n[ANALYSIS 2] Segmenting trips by type...")
df_segmented = df_enriched.withColumn(
    "trip_type", 
    when((col("trip_distance") < 1) & (col("total_amount") < 10), "Short Local")
    .when((col("trip_distance") >= 1) & (col("trip_distance") < 5) & (col("total_amount") < 20), "Medium Local")
    .when((col("trip_distance") >= 5) & (col("trip_distance") < 15), "Long Distance")
    .otherwise("Very Long")
)
segment_analysis = df_segmented.groupBy("trip_type").agg(
    count("*").alias("trip_count"),
    avg("total_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance")
).orderBy(desc("trip_count"))

print("Trip Segmentation:")
segment_analysis.show()
print("\n[ANALYSIS 3] Analyzing temporal patterns...")
df_temporal = df_enriched.withColumn(
    "time_period",
    when((hour("pickup_time") >= 5) & (hour("pickup_time") < 12), "Morning")
    .when((hour("pickup_time") >= 12) & (hour("pickup_time") < 17), "Afternoon")
    .when((hour("pickup_time") >= 17) & (hour("pickup_time") < 21), "Evening")
    .otherwise("Night")
)
temporal_summary = df_temporal.groupBy("time_period").agg(
    count("*").alias("trips"),
    avg("total_amount").alias("avg_fare"),
    stddev("total_amount").alias("fare_std_dev"),
    max("total_amount").alias("max_fare")
).orderBy("time_period")
print("Temporal Analysis:")
temporal_summary.show()
print("\n[ANALYSIS 4] Comparing payment methods...")

payment_analysis = df_enriched.groupBy("payment_type").agg(
    count("*").alias("transaction_count"),
    avg("total_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance")
)
print("Payment Method Comparison:")
payment_analysis.show()
print("\n[ANALYSIS 5] Analyzing distance patterns...")

df_distance_buckets = df_enriched.withColumn(
    "distance_bucket",
    when(col("trip_distance") <= 1, "0-1 mi")
    .when(col("trip_distance") <= 2, "1-2 mi")
    .when(col("trip_distance") <= 5, "2-5 mi")
    .when(col("trip_distance") <= 10, "5-10 mi")
    .otherwise("10+ mi")
)
distance_summary = df_distance_buckets.groupBy("distance_bucket").agg(
    count("*").alias("trip_count"),
    avg("total_amount").alias("avg_fare")
).orderBy("distance_bucket")

print("Distance Distribution:")
distance_summary.show()
print("\n[ANALYSIS 6] Identifying peak hours...")

hourly_demand = df_enriched.groupBy("hour_of_day").agg(
    count("*").alias("trip_count"),
    avg("total_amount").alias("avg_fare")
).orderBy(desc("trip_count"))

print("Top 10 Busiest Hours:")
hourly_demand.show(10)
print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print("="*70)
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print("\n[1] AGGREGATED SUMMARY (168 hour-day combinations):")
df_summary.show(20)
print("\n[2] ANOMALIES (281K unusual trips):")
anomalies.select("hour_of_day", "total_amount", "trip_distance").show(10)
print("\n[3] TRIP SEGMENTATION:")
segment_analysis.show()
print("\n[4] TEMPORAL PATTERNS:")
temporal_summary.show()
print("\n[5] PAYMENT METHODS:")
payment_analysis.show()
print("\n[6] DISTANCE DISTRIBUTION:")
distance_summary.show()
print("\n[7] PEAK HOURS:")
hourly_demand.show(10)
