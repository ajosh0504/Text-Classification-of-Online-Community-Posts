# Text Classification of Online Community Posts
This project uses posts from Breastcancer.org, a popular peer-to-peer online health community(OHC) for breast cancer survivors and caregivers. The dataset consists of all public posts and user profile information available on the community from October 2002 to August 2013. There are more than 2.8 million posts, contributed by 49,552 users.

There are five major social support activities on the OHC- Companionship(COM), Providing Emotional Support(PES), Providing Informational Support(PIS) and Seeking Emotional/Informational Support(SS). The task is to classify each of the posts into their respective categories.

The data used in the project cannot be made public for reasons of confidentiality.

# Order of execution of files
1) Generate_Gensim_Model_files.py - Generate model for word emebedding
2) Compute_Word_Embeddings.py - Generate word embeddings
3) 10_fold_classifier_and_predict_on_all.py - Trains a Random Forest Classification Model on a pre-annotated dataset and predicts labels for our actual dataset.
