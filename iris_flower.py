import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import evaluate
import torch

'''
    x = iris.data is a 2D array where each x represents sepal length, sepal width, petal length and petal width (cm)
    y = iris.target is an array where each y represents species; 0 - setosa, 1 - versicolor, 2 - virginica
    X = represents a training dataset btw
    
    have an llm predict a y value based off an x array using knn techniques
    compare llm predictions to linear regression predictions to evaluate the model's ability to perform classification
    finetune llm off of scikitlearn datasets
    using knn, if two results are very close to the point, have a random number generator handle the rest
    incorporate np.random for ranges
    use eval() to convert string input to array
    store model predictions in a vector database
    tokenize predictions to determine cosine or euclidean similarity between new and old predictions
    include metrics
    then try to implement computer vision after
'''

dataset = load_dataset('scikit-learn/iris')
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors = "pt")
    outputs = model.generate(**inputs)
    completion = (tokenizer.batch_decode(outputs, skip_special_tokens = True))
    return completion[0]
def tokenize(query): # for text
    return -1
    inputs = tokenizer(query, return_tensors = "pt")
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(**inputs)
    encoder_embeddings = encoder_outputs.last_hidden_state
    return encoder_embeddings
def iris_graph():
    iris = load_iris()
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")
    plt.show()
def knn_instance():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    knn = KNeighborsClassifier(5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)
def fine_tune():
    dataset['train'][100]

    def tokenize_function(examples):
        return tokenizer(examples["Species"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(120))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(30))
    training_args = Seq2SeqTrainingArguments("test-trainer")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return metric.compute(predictions = predictions, references = labels)
    
    trainer = Seq2SeqTrainer(
        model = model, 
        args = training_args, 
        train_dataset = small_train_dataset, 
        eval_dataset = small_eval_dataset, 
        compute_metrics = compute_metrics
    )

    trainer.train()

print(knn_instance())
