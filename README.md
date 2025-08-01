## Large Transaction Models

The model architecture and corresponding code in this repository is largely unchanged from the FATA-Trans github, with some minor changes to fix bugs that I encountered when adapting the codebase to my dataset. The sort of entry point for running the code in this repository is the "single_experiment.sh" shell script in the exp_components/ subdirectory. I will try to break down everything that happens when you run that script (or everything that _can_ happen depending on the arguments):

### preprocess.py

Given the choices of --dataset, --static_features, --include_time_features, --include_user_features,  and --include_market_features, the script "preprocess.py" will load the appropriate dataset from the IDEA cluster (if it exists) and do some preprocessing of the data. After the preprocessing, the data is saved in "/data/IDEA_Defi_Research/Data/[dataset_specific_path]/preprocessed_[feature_sets]/". The preprocessing can take a few minutes or longer, and so for convenience, you can also include the argument "--check_preprocess_cached" and the preprocess script will check to see if there is already data where it would intend to write its preprocessed data should it go through the whole script, and if so, it assumes this exact dataset has already been preprocessed. I recommend always including this argument unless you make changes to the preprocess.py script.

### preload.py

This is where the vocabulary objects are created and the columns are encoded based on the --num_bins argument. It loads data from the appropriate preprocessed/ directory and basically uses the dataset/preload.py dataset class to create the encoders and vocab objects for the data. There is also a "--check_preload_cached" argument you can pass to avoid redoing this process every time if you're using the same vocab and dataset for repeated runs.

### process.py

I'm not sure this file is necessary at all, honestly, but in theory I think the intent of the "process.py" script is to run the preloaded data through a collator to create the transaction sequence samples for use in the pretraining phase. However, the caching of the "user samples" doesn't seem to work in the pretraining phase, so I think this basically just creates the user samples and can be useful to confirm that things are working properly before running the full pretrain.

### pretrain.py

As the name suggests, this is where the pretraining of the model actually happens. This is a pared-down version of FATA-Trans's main.py. I limited the options for models and data collators and whatnot to just a couple, because I didn't and still don't intend to ablation test all of the FATA-Trans novelties. They already did so. But this basically re-does the user-sample creation that happens in "process.py" and then sets up the model and trains it based on the various choices of arguments.

## Using the pre-trained model:
Assuming you've just successfully trained a model using the "single_experiment.sh" script and now you want to export the embeddings for that same dataset for use in some downstream task, you should be able to do this easily by copy-pasting all the arguments used in "single_experiment.sh" into "single_model_export.sh" and changing the --checkpoint argument to -1, which will cause it to load the final_model checkpoint (or just the checkpoint with the highest number, if final_model doesn't exist for some reason), and generate the embeddings for the same data that was used to train the model. These embeddings are saved in the same directory with the model checkpoint that was used to generate them.

The only task that I've actually properly done any evaluation with is the "Survival_Prediction" task. To use the embeddings, go to the exp_components/Survival_Prediction/survivalPrediction.ipynb file, set the data_path appropriately to point to the cls_embeddings you generated, and run the file. It should write the embeddings as a .csv file to your home directory. This is where things get a bit messy, because all the code to run the survival prediction and classification tasks was originally written in R, and so we basically need to get these embeddings into the same structure that was used for the existing pipelines in order to evaluate their effectiveness.

[Incomplete... still working on this]

