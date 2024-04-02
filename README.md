# How to run code

## 1. Create Conda Environment:

**conda env create -f environment.yml**

## 2. Activate Conda Environment:

**conda activate env**

## 3. Fine-Tune a model:

**Fine-tuning models are done with traning_script.py**
In this script users are able to chage the base_model variable with a HuggingFace Repo \n
to perform instruction fine-tuning on the Alpaca instruction datset (currently available in alpaca_data.json)

## Discussion:

**Fine tuning discussion**
The initial results of fine-tuning do not indicate a strong response. My hypothesis is the model was under trained during the fine tuning process due to hardware limitations.
Future work will include exploring other quantization methods, other libraries such as DeepSpeed and Unsloth, and continuing to learn more about LoRA, QLoRA, and PeFT.

The metrics used to evaluate the models explored in this work are bilingual evaluation understudy (BLEU), Recall-Oriented Understudy for Gisting Evaluation (ROUGE) - Longest Common Subsequence (LCS), and BERTScore.

- BLEU is a widely used metric for evaluating generated text. It is important to note that this and the proceeding metrics do require a *ground truth* reference. At its core, BLEU determines the precision of contiguous sequences of text - known as N-Grams. The calculation for this looks at a 1-word, 2-word, 3-word, and 4-word sequence of words in the generated text and then compares it to the ground truth to evaluate the precision of the generated text. These scores can range from 0 to 1, where 1 means perfect overlap with the ground truth text.

- ROUGE-L is primarily used for evaluating summarization capabilities of text generation models. The focus being on the LCS of the sequence w.r.t the ground truth text. It is worth noting the results for ROUGE-L, as depicted in the below figure, is the relation between precision and recall - otherwise known as the F1-score.

- BERTScore, the most relevant for the instruction tuned data generation explored in this work, uses contextual embeddings to evaluate semantic similiarity between the generated text and ground truth text. The adaptability of this particular metric is its independence from N-Gram matching in sequence generation. Again, as was the case with ROUGE-L, F1-score was used as a measure of success.

**Hyperparameter Discussion**
The following hyperparameter evaluations were perfomed using the inference.py script. The script is currently being used for human-evaluations, but with slight modifications (and working within appropriate resources) it can be expanded to include list containing parameters of interest being fed into the output generator.

- Top K is a sampling method to intrduce randomness. Where the associated 'k' value dictates the the size of the subset used to sample from. The subset, characterized by 'k', is a list of the top most probable word choices (again, of size k) that is then randomly selected for the final word choce. This allows for more deterministic generation patterns with lower k values (i.e. reduce the size of the sampling space to only those with high probabilities) or a more creative generation with higher values of k (i.e. expand the sampling space to include words with both high and lower probabilities).

- Beam Size is an optimization strategy for generating text. The beam size is the number of steps a model tracks when trying to maximize the output probability for a generated sequence of text. A larger beam size can lead to a more coherent output. However, this comes with a trade off due to the number of computations increasing drastically the further you expand the search space (increase beam size).

- Temperature is a hyper prameter added to the softmax activation function. Whos impact on generation is determined by the modification to the underlying output probabilities of text selection (and resulting text generation). A lower temperature filters the output probabilities to only include the highest probability of world selection and a value of 1 indicates no change to the distribution sampled from during the world selection process. It is worth noting, lower values are better suited for applications who values need to be more deterministic and resiliant to noise or fanciful wording.


#TODO - Add base model performance to the results.png

![Results](assets/results.png "Default evaluation on instruction tuned data with addtional evaluations performed on varying hyperparameters")