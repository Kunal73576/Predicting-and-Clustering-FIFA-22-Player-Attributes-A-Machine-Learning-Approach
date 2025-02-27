from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, desc
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize Spark session
spark = SparkSession.builder.appName("FIFA22Analysis").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # Set Spark log level to ERROR

# Load the data
data_path = './data.csv'
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Define the desired features, ensuring they match available columns
desired_features = [
    'long_name', 'potential', 'value_eur', 'wage_eur', 'age', 'height_cm', 'weight_kg',
    'international_reputation', 'weak_foot', 'skill_moves', 'work_rate', 'body_type',
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration',
    'vision', 'penalties', 'composure', 'stamina', 'strength'
]

# Select only the available desired features
selected_features = [feature for feature in desired_features if feature in data.columns]

data = data.select(selected_features)

# Handle missing values
data = data.na.fill(0)

# Calculate average attacking score
attacking_columns = [
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration'
]

data = data.withColumn('average_attacking', sum([col(c) for c in attacking_columns]) / len(attacking_columns))

# Index the categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data) for column in ['work_rate', 'body_type']]
for indexer in indexers:
    data = indexer.transform(data)

# Update selected features to include indexed columns
selected_features = [col+"_index" if col in ['work_rate', 'body_type'] else col for col in selected_features if col != 'long_name']

# Assemble features into a feature vector
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
assembled_data = assembler.transform(data)

# Split the data into training and test sets, including long_name column separately
train_data, test_data = assembled_data.randomSplit([0.7, 0.3], seed=42)
train_data = train_data.select("long_name", "features", "average_attacking")
test_data = test_data.select("long_name", "features", "average_attacking")

# Function to evaluate model
def evaluate_model(train_data, test_data, features_col, label_col="average_attacking"):
    start_time = time.time()
    dt = DecisionTreeRegressor(featuresCol=features_col, labelCol=label_col)
    dt_model = dt.fit(train_data)
    training_time = time.time() - start_time

  # Save the trained model with overwrite option
    model_save_path = "./saved_models/decision_tree_model"
    dt_model.write().overwrite().save(model_save_path)
    print(f"Model saved at {model_save_path}")

    # Training predictions and metrics
    train_predictions = dt_model.transform(train_data)
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    train_rmse = evaluator_rmse.evaluate(train_predictions)
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    train_r2 = evaluator_r2.evaluate(train_predictions)

    # Testing predictions and metrics
    test_predictions = dt_model.transform(test_data)
    test_rmse = evaluator_rmse.evaluate(test_predictions)
    test_r2 = evaluator_r2.evaluate(test_predictions)

    return train_predictions, test_predictions, train_rmse, test_rmse, train_r2, test_r2, training_time

# 1. Without Scaling
train_predictions_no_scaling, test_predictions_no_scaling, train_rmse_no_scaling, test_rmse_no_scaling, train_r2_no_scaling, test_r2_no_scaling, time_no_scaling = evaluate_model(train_data, test_data, features_col="features")
print(f"-------------------------------------------------------")
print(f"Training RMSE without scaling: {train_rmse_no_scaling}")
print(f"Testing RMSE without scaling: {test_rmse_no_scaling}")
print(f"Training R^2 without scaling: {train_r2_no_scaling}")
print(f"Testing R^2 without scaling: {test_r2_no_scaling}")
print(f"Training time without scaling: {time_no_scaling} seconds")
print(f"-------------------------------------------------------")

# 2. With Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)
train_data_scaled = scaled_data.select("long_name", "scaledFeatures", "average_attacking").withColumnRenamed("scaledFeatures", "features")
test_data_scaled = scaled_data.select("long_name", "scaledFeatures", "average_attacking").withColumnRenamed("scaledFeatures", "features")
train_predictions_with_scaling, test_predictions_with_scaling, train_rmse_with_scaling, test_rmse_with_scaling, train_r2_with_scaling, test_r2_with_scaling, time_with_scaling = evaluate_model(train_data_scaled, test_data_scaled, features_col="features")
print(f"-------------------------------------------------------")
print(f"Training RMSE with scaling: {train_rmse_with_scaling}")
print(f"Testing RMSE with scaling: {test_rmse_with_scaling}")
print(f"Training R^2 with scaling: {train_r2_with_scaling}")
print(f"Testing R^2 with scaling: {test_r2_with_scaling}")
print(f"Training time with scaling: {time_with_scaling} seconds")
print(f"-------------------------------------------------------")

# 3. With Scaling and PCA
pca = PCA(k=10, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(train_data_scaled)
pca_data = pca_model.transform(train_data_scaled).select("long_name", "pcaFeatures", "average_attacking").withColumnRenamed("pcaFeatures", "features")
pca_test_data = pca_model.transform(test_data_scaled).select("long_name", "pcaFeatures", "average_attacking").withColumnRenamed("pcaFeatures", "features")
train_predictions_with_pca, test_predictions_with_pca, train_rmse_with_pca, test_rmse_with_pca, train_r2_with_pca, test_r2_with_pca, time_with_pca = evaluate_model(pca_data, pca_test_data, features_col="features")
print(f"-------------------------------------------------------")
print(f"Training RMSE with PCA: {train_rmse_with_pca}")
print(f"Testing RMSE with PCA: {test_rmse_with_pca}")
print(f"Training R^2 with PCA: {train_r2_with_pca}")
print(f"Testing R^2 with PCA: {test_r2_with_pca}")
print(f"Training time with PCA: {time_with_pca} seconds")
print(f"-------------------------------------------------------")

# 4. With PCA but Without Scaling
pca_without_scaling = PCA(k=10, inputCol="features", outputCol="pcaFeatures")
pca_model_without_scaling = pca_without_scaling.fit(train_data)
pca_data_without_scaling = pca_model_without_scaling.transform(train_data).select("long_name", "pcaFeatures", "average_attacking").withColumnRenamed("pcaFeatures", "features")
pca_test_data_without_scaling = pca_model_without_scaling.transform(test_data).select("long_name", "pcaFeatures", "average_attacking").withColumnRenamed("pcaFeatures", "features")
train_predictions_pca_without_scaling, test_predictions_pca_without_scaling, train_rmse_pca_without_scaling, test_rmse_pca_without_scaling, train_r2_pca_without_scaling, test_r2_pca_without_scaling, time_pca_without_scaling = evaluate_model(pca_data_without_scaling, pca_test_data_without_scaling, features_col="features")
print(f"-------------------------------------------------------")
print(f"Training RMSE with PCA but without scaling: {train_rmse_pca_without_scaling}")
print(f"Testing RMSE with PCA but without scaling: {test_rmse_pca_without_scaling}")
print(f"Training R^2 with PCA but without scaling: {train_r2_pca_without_scaling}")
print(f"Testing R^2 with PCA but without scaling: {test_r2_pca_without_scaling}")
print(f"Training time with PCA but without scaling: {time_pca_without_scaling} seconds")
print(f"-------------------------------------------------------")

# Save the results to CSV files
train_predictions_with_pca.select("long_name", "average_attacking", "prediction").write.csv('./train_predictions_pca_without_scaling.csv', header=True)
test_predictions_with_pca.select("long_name", "average_attacking", "prediction").write.csv('./test_predictions_pca_without_scaling.csv', header=True)

# Enhanced Plotting with Seaborn
approaches = ['No Scaling', 'With Scaling', 'With PCA', 'PCA without Scaling']
train_rmse = [train_rmse_no_scaling, train_rmse_with_scaling, train_rmse_with_pca, train_rmse_pca_without_scaling]
test_rmse = [test_rmse_no_scaling, test_rmse_with_scaling, test_rmse_with_pca, test_rmse_pca_without_scaling]
train_r2 = [train_r2_no_scaling, train_r2_with_scaling, train_r2_with_pca, train_r2_pca_without_scaling]
test_r2 = [test_r2_no_scaling, test_r2_with_scaling, test_r2_with_pca, test_r2_pca_without_scaling]

plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

# Plotting RMSE
plt.subplot(1, 2, 1)
sns.lineplot(x=approaches, y=train_rmse, marker='o', label='Train RMSE', color='b')
sns.lineplot(x=approaches, y=test_rmse, marker='o', label='Test RMSE', color='r')
plt.xlabel('Approach')
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.legend()

plt.subplot(1, 2, 2)
sns.lineplot(x=approaches, y=train_r2, marker='o', label='Train R^2', color='b')
sns.lineplot(x=approaches, y=test_r2, marker='o', label='Test R^2', color='r')
plt.xlabel('Approach')
plt.ylabel('R^2')
plt.title('R^2 Comparison')
plt.legend()

plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.show()

##### Perform K-Means Clustering
kmeans = KMeans(k=5, featuresCol="features", predictionCol="prediction")  # Adjust k as needed
model = kmeans.fit(pca_data)
predictions = model.transform(pca_data)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette with PCA: {silhouette}")

# Sort players by average attacking score in descending order
sorted_predictions = predictions.orderBy(desc("average_attacking"))

# Display the results in a more readable manner
sorted_predictions.select("long_name", "average_attacking", "prediction").show(10, truncate=False)

# Save the results to a CSV file
sorted_predictions.select("long_name", "average_attacking", "prediction").write.csv('./Cluster_predictions_with_pca.csv', header=True)

spark.stop()

