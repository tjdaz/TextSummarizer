# TextSummarizer
Generates summaries for the given input text.

## Setup/Installation
This program was designed to be used in Python 3.11+.

Clone the project and install the requirements using pip in your virtual environment:

```bash
pip3 install -r requirements.txt
```

Download the compressed project data folder:
https://drive.google.com/file/d/1rmDwRwekZJDIE9GHPFGcuVLM9XqvTU-9/view?usp=sharing

This contains a small portion of the CNN/Daily Mail dataset, the GloVe 300d text file, and a pickled embedding index.

If you would like to avoid building the database and model, you can download the pre-built data folder:  
https://placeholder

This is a small model trained on a small set of articles and summaries.

Store the un-built or pre-built data folder in the main project directory path:
```bash
TextSummarizer/data
```

The data directory structure should look like this:
```
TextSummarizer/
— data/
  — database/
  — glove/
  — models/
  — raw/
```

## Using TextSummarizer

The build.py file should be run if you have downloaded the un-built data folder.
```bash
python build.py
```

This will pre-process the data, store it in the database, train the neural network, and run ROUGE metrics on it.

The model trains on a very small dataset here. Several constants can be changed in neural_network.py to adjust training.

After building, or if you have downloaded the pre-built data folder, you can generate summaries using summarizer.py. It takes one argument, the path to the file you want to summarize. This file must only contain plain text. The summarizer can be run on any of the files in data/raw to get an example summary.

```bash
python summarizer.py data/raw/example_file.story
```

The summary will be generated and printed to the console.

The metrics.py file can be run to get metrics on a file in the database. An ID must be entered into the main() function to get its metrics.


