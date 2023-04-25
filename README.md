# TextSummarizer
Generates summaries for the given input text.

## How It Works
Placeholder.

## Setup/Installation
This program was designed to be used in Python 3.11+.

Clone the project and install the requirements using pip in your virtual environment:

```bash
pip3 install -r requirements.txt
```

Download the CNN portion of the CNN/Daily Mail dataset:  
https://drive.google.com/file/d/1lCq4b9CX_GH8g8YB4yEBary47yKPQljM/view?usp=sharing

Store the files in the data path:
```bash
data/raw
```

If you are using a NVIDIA GPU and have CUDA setup on your system, you can run the following command to install CUDA supported versions of the torch packages:
```bash
pip3 install torch==2.0.0+cu118 torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118
````

If not, processing will be done on your CPU instead (this will take a long time).

If you would like to avoid downloading and processing the CNN files to build the database, you can download the database we built for the program:  
https://drive.google.com/file/d/1jXMAVx86iF_VOJoh3DzVn7rLO2R8n0_0/view?usp=share_link

Store the database in the data path:
```bash
data/database
```

The data directory structure should look like this:
```
TextSummarizer/
— data/
  — database/
— models/
— raw/
  — test/
  — train/
  — val/
```