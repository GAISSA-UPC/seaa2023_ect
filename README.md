# seaa2023_ect
Replication package for the paper "Do DL models and training environments have an impact on the energy consumption?".

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7828519.svg)](https://doi.org/10.5281/zenodo.7828519)

## Set up the environment
### Installing dependencies
Before executing the code, you must first install the required dependencies.
We use [Poetry](https://python-poetry.org/docs/) to manage the dependencies.

If you want to use any other dependency manager, you can look at the [pyproject.toml](pyproject.toml) file for the required dependencies.
 
### MLflow configuration
We use [MLflow](https://mlflow.org/docs/latest/index.html) to keep track of the different experiments. By default, its usage
is disabled. If you want to use MLflow, you need to:
- Configure your own [tracking server](https://mlflow.org/docs/latest/tracking.html#tracking-server).
- Activate MLflow logging in the [experiment.yaml](config/experiment.yaml) configuration file.
- Create a `secrets.yaml` file in the project root with the following structure:
```yaml
MLFLOW:
  URL: https://url/to/your/tracking/server
  USERNAME: tracking_server_username
  PASSWORD: tracking_server_password
```
If you do not need users credentials for MLflow leave the fields empty.

### Resources configuration
This package uses a resources configuration to manage the GPU memory limit allowed and the use of cache.
We do not share this file since each machine has its own hardware specifications.
You will need to create a `resources.yaml` file inside the `config` folder with the following structure:

```yaml
GPU:
  MEM_LIMIT: 2048

USE_CACHE: true

```
The GPU memory limit must be specified in Megabytes. If you do not want to set a GPU memory limit, leave the field empty.

__WARNING! The memory limit value is just an example. Do not take it as a reference.__

## Running the experiment
Once the environment is set up, you can run the experiment by executing the following command:

```console
$ python -m src.profiling.profile_model [--experiment_name EXPERIMENT_NAME] [-d DATA] {local, cloud}

positional arguments:
  {local,cloud}         The type of training environment.

options:
  --experiment_name EXPERIMENT_NAME
                        The name of the MLflow experiment.
  -d DATA, --data DATA  Path to the dataset folder. The default is the data/dataset folder.
```

The raw measurements for each architecture will be saved in the `data/metrics/raw/{local, cloud}/architecture_name` folder.
If MLflow is enabled, the measurements will also be saved in the MLflow tracking server, together with the trained models.
If not, the trained models will be saved in the `models` folder and the training history will be saved with the raw measurements as `performance-%Y%m%dT%H%M%S.csv`.

You can also train a single model without profiling the energetic metrics by executing the following command:

```console
$ python -m src.models.run_training [--experiment_name EXPERIMENT_NAME] [-d DATA] {local,cloud} {vgg16,resnet50,xception,mobilenet_v2,nasnet_mobile}

positional arguments:
  {local,cloud}         Whether is to be executed locally or in the cloud.
  {vgg16,resnet50,xception,mobilenet_v2,nasnet_mobile}
                        Architecture of the DNN

options:
  --experiment_name EXPERIMENT_NAME
                        The name of the MLflow experiment.
  -d DATA, --data DATA  Path to the dataset folder. The default is the data/dataset folder.
```

The training history and the model will be saved following the same rules as the profiling script.

### Training data
We do not share the training data used in this experiment. However, you can use any dataset you want, as long as it is
intended for binary image classification, and obtain the energy measurements for it.

## Collected data
All the data collected during the experiment can be found in the [data/metrics](data/metrics) folder. The data is organized in the following structure:

.  
├── - raw  
├── - interim  
└── - processed

The `raw` folder contains the raw measurements collected during the experiment.
The `interim` folder contains the processed data that is used to generate the final dataset.
The `processed` folder contains the final data used to perform the analysis.

## Data analysis
The data analysis is done using [Jupyter Notebooks](https://jupyter.org/). You can find the analysis in the [data-analysis.ipynb](data-analysis.ipynb) file. All the plots generated are saved in the `out/figures` folder.

## License
The software under this project is licensed under the terms of the Apache 2.0 license. See the [LICENSE](LICENSE) file for more information.

The data used in this project is licensed under the terms of the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. See the [LICENSE](data/LICENSE) file for more information.
