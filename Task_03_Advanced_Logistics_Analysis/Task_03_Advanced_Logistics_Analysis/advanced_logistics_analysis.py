# Task 03 - Advanced Data Analysis and Visualization in Logistics
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/advanced_logistics_dataset.csv", parse_dates=["order_date"])

print(df.describe(include="all"))

kpis = {
    "Average Delivery Days": df["actual_delivery_days"].mean(),
    "Median Delivery Days": df["actual_delivery_days"].median(),
    "On-Time Delivery Rate (%)": df["on_time_delivery"].eq("Yes").mean() * 100,
    "Average Total Cost": df["total_cost"].mean(),
    "Average Customer Rating": df["customer_rating"].mean()
}
print("\nKPIs:")
for key, value in kpis.items():
    print(f"{key}: {value:.2f}")

# Delivery time distribution
plt.figure(figsize=(8,5))
plt.hist(df["actual_delivery_days"], bins=30)
plt.title("Distribution of Actual Delivery Time")
plt.xlabel("Actual Delivery Days")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()

# Monthly volume
monthly = df.groupby(df["order_date"].dt.to_period("M"))["shipment_id"].count()
plt.figure(figsize=(9,5))
plt.plot(monthly.index.astype(str), monthly.values, marker="o")
plt.title("Monthly Shipment Volume")
plt.xlabel("Month")
plt.ylabel("Number of Shipments")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cost by mode
mode_cost = df.groupby("transport_mode")["total_cost"].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
plt.bar(mode_cost.index, mode_cost.values)
plt.title("Average Logistics Cost by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Average Total Cost")
plt.tight_layout()
plt.show()

# Regional on-time rate
region_ontime = df.groupby("region")["on_time_delivery"].apply(lambda x: x.eq("Yes").mean()*100)
plt.figure(figsize=(8,5))
plt.bar(region_ontime.index, region_ontime.values)
plt.title("On-Time Delivery Rate by Region")
plt.xlabel("Region")
plt.ylabel("On-Time Rate (%)")
plt.ylim(0,100)
plt.tight_layout()
plt.show()

# Distance vs cost
plt.figure(figsize=(8,5))
plt.scatter(df["distance_km"], df["total_cost"], alpha=0.35)
plt.title("Distance vs Total Logistics Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Total Cost")
plt.tight_layout()
plt.show()

print("\nDistance-cost correlation:",
      round(df["distance_km"].corr(df["total_cost"]), 3))

print("\nCorrelation matrix:")
print(df[[
    "shipment_volume_kg","distance_km","estimated_delivery_days",
    "actual_delivery_days","delivery_delay_days",
    "transportation_cost","handling_cost","fuel_surcharge",
    "total_cost","customer_rating"
]].corr().round(2))
