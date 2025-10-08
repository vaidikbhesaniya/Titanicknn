# API Documentation

## Data Processing Module (`src.data.preprocessing`)

### `load_raw_data(file_path: Path) -> pd.DataFrame`
Loads the raw Titanic dataset from CSV file.

**Parameters:**
- `file_path`: Path to the CSV file

**Returns:**
- `pd.DataFrame`: Loaded dataset

### `drop_unnecessary_columns(data: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame`
Removes specified columns from the dataset.

**Parameters:**
- `data`: Input DataFrame
- `columns_to_drop`: List of column names to remove

**Returns:**
- `pd.DataFrame`: Dataset with specified columns removed

## Feature Engineering Module (`src.features.engineering`)

### `chi_square_feature_selection(data: pd.DataFrame, target_column: str, significance_threshold: float) -> List[str]`
Performs chi-square test for feature selection.

**Parameters:**
- `data`: DataFrame with features and target
- `target_column`: Name of the target column
- `significance_threshold`: P-value threshold for significance

**Returns:**
- `List[str]`: List of statistically significant feature names

## Training Module (`src.training.train`)

### `train_knn_model(X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int) -> KNeighborsClassifier`
Trains a K-Nearest Neighbors classifier.

**Parameters:**
- `X_train`: Training feature array
- `y_train`: Training target array
- `n_neighbors`: Number of neighbors for KNN

**Returns:**
- `KNeighborsClassifier`: Trained model

## Prediction Module (`src.prediction.predictor`)

### `TitanicPredictor`
Main class for making predictions on new Titanic passenger data.

#### `predict_single(passenger_data: Dict) -> Dict`
Makes prediction for a single passenger.

**Parameters:**
- `passenger_data`: Dictionary with passenger information

**Returns:**
- `Dict`: Prediction results with probabilities

#### `predict(data: Union[pd.DataFrame, Dict]) -> np.ndarray`
Makes predictions on batch data.

**Parameters:**
- `data`: New data to predict (DataFrame or dict)

**Returns:**
- `np.ndarray`: Predictions array

## Configuration (`src.config.settings`)

Contains all project configuration including:
- File paths
- Model parameters
- Feature specifications
- Default settings