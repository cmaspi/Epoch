# Epoch Core Selection Hackathon
## AI20BTECH11006

### Results

### Preprocessing
I used two kinds of pre-processing, one being PCA which I used for KNN and Logistic Regression. The other being just removing two features (l, m value) which was used for Decision Tree and Random forest. I used this method because it gave me slightly higher accuracy and it doesn't lead to loss in explanability in case of Decision Tree.

### Classifiers Used
1. KNN
2. Logistic Regression
3. Decision Tree
4. Random Forest
> Further in case of Random forest, since the dataset was too small I eneded up using a higher fraction of data and features for each tree

On top of these models, stacking was done to get slightly more stable results.

-----------

Find the relevant code for classifiers in `src` directory 

