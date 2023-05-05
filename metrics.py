"""
Name: Timothy James Duffy, Kevin Falconett
File: metrics.py
Class: CSc 483; Spring 2023
Project: TextSummarizer

Provides methods to calculate the ROUGE metric and print the results.
"""

# Filter tensorflow warnings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from database import *
from rouge import Rouge
from config import DATABASE_NAME
from summarizer import generate_summary


def get_rouge_scores(num_docs, offset):
    """Gets the rogue scores using documents in the database. Documents can be offset by id."""
    # Get articles and summaries from the database.
    database = Database(DATABASE_NAME)
    articles, summaries = zip(*database.get_data(num_docs, offset))

    # Holds the generated summaries.
    generated_summaries = []

    # Generate summaries for the articles.
    for i, article in enumerate(articles):
        summary = generate_summary(article)
        generated_summaries.append(summary)
        print('Actual summary:\n{}'.format(summaries[i]))
        print('Generated summary:\n{}\n'.format(generated_summaries[i]))

    # Calculate ROUGE scores for the sample set.
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, list(summaries), avg=True)
    return scores


def print_scores(scores):
    """Prints the given ROUGE scores in a nice format."""
    # Get ROUGE dictionaries. Each contains recall, precision, accuracy scores.
    r1 = scores['rouge-1']
    r2 = scores['rouge-2']
    rl = scores['rouge-l']

    # Print out the scores for Rouge-1, Rouge-2, and Rouge-l.
    print('Rouge-1\trecall:\t{:.2f}\tprecision:\t{:.2f}\tf1_score:\t{:.2f}'.format(r1['r'], r1['p'], r1['f']))
    print('Rouge-2\trecall:\t{:.2f}\tprecision:\t{:.2f}\tf1_score:\t{:.2f}'.format(r2['r'], r2['p'], r2['f']))
    print('Rouge-l\trecall:\t{:.2f}\tprecision:\t{:.2f}\tf1_score:\t{:.2f}\n'.format(rl['r'], rl['p'], rl['f']))


def main():
    # Prints the ROUGE results for data the model has been trained on.
    print('Trained Data ROUGE Scores:')
    trained_data = get_rouge_scores(1, 0)
    print_scores(trained_data)

    # Prints the ROUGE results for data the model has NOT been trained on.
    print('Untrained Data ROUGE Scores:')
    untrained_data = get_rouge_scores(1, 1200)
    print_scores(untrained_data)


if __name__ == '__main__':
    main()
