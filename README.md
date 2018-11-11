## Introduction

Fine grain sentiment analysis from AI challenger, with data and runnable code and performances

Important Notice: this project is based on pengshuang/AI-Comp's project(baseline), most of code is reused here. 

the main reason why we reused this baseline is to save time as this baseline is runnable and has a reasonable performance 

already. Online f1 score is 0.702 as reported in this baseline. since we just want to try some new ideas, to do some 

improvement, for example, we want to add some new models, and use some new techniques introduced recently such as 

pre-train of language model on large corpus and gain fine. this project is on its early stage.

<a href='https://challenger.ai/competition/fsauor2018'>Introduction to fine grain sentiment from AI Challenger</a>


## Experiment on New Models

add something here.
      
   <img src="https://github.com/brightmart/sentiment_analysis_fine_grain/blob/master/data/img/fine_grain.jpg"  width="65%" height="65%" />

   for more, check model/bert_cnn_fine_grain_model.py

## Usage
    
     1. download <a href='https://pan.baidu.com/s/19aMHbPgfpBxz9sS-sYsjOg'>cache file of sentiment analysis</a>
     
     2. train the model:
        
        python train_cnn_fine_grain.py
 
 # Dataset and Cahce file
  
 cache file was generate by following steps from preprocess_word.ipynb. it contains everything you need for this task, 
 
 including: processed train/validation/test set; vocabulary of word; a dict map label to index. 
 
 take train_valid_test_vocab_cache.pik and put it under folder of preprocess_word/
 
 raw data are also included in this zip file.
 

       
## OLD Usage

    1. generate train/validation/test set(char level:
       
       Preprocess_char.ipynb
    
    2. train the model:
       
       python model_*_char.py to 
    
    3. make prediction using validation data, and write prediction to file
    
       python validation_*_char.py
    
    4. compute f1 score on validation set
     
       python evaluate_char.py
      
    5. submmit prediction on test set, and generate submition file 
    
       python predict_*_char.py
    

    class_*.py file for model
    model_*_char.py file for training
    validation_*_char.py genreate validation result  
    evaluate_char.py compute f1 score on validation set 
    predict_*_char.py generate online submit file 
    
## Reference

1. <a href='https://github.com/pengshuang/AI-Comp'>pengshuang/AI-Comp</a>

2. <a href='https://github.com/AIChallenger/AI_Challenger_2018'>AI Challenger 2018</a>