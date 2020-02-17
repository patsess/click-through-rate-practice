
"""
Note: code is for reference only (taken from an online course)
"""


if __name__ == '__main__':
    # Precision and recall:

    # Precision: ROI on ad spend through clicks
    # - Low precision means very little tangible ROI on clicks

    # Recall:targeting relevant audience
    # - Low recall means missed out opportunities on ROI

    # It may be sensible to weight the two differently
    # - Companies are likely to care more about avoiding low precision
    # compared to low recall

    ######################################################################
    # F-beta score:

    # F = (1 + beta) x [(precision x recall) / ([beta^2 x precision] + recall)]

    # Beta coefcient: represents relative weighting of two metrics
    # - Beta between 0 and 1 means precision is made smaller and hence
    # weighted more, whereas beta > 1 means precision is made larger and hence
    # weighted less

    # Implementation available in sklearn via: fbeta_score(y_true, y_pred,
    # beta)
    # - y_true is true targets and y_pred the predicted targets

    ######################################################################
    # AUC of ROC curve versus precision:

    # roc_auc = roc_auc_score(y_test, y_score[:, 1])

    # fpr = 1 - tn / (tn + fp)
    # precision = tp / (tp + fp)

    # - Imbalanced dataset: fpr can be low when precision is also low.
    # - Let us assume we have 100 TN, and 10 TP and 10 FP.

    # fpr = 1 - 100 / (100 + 10) = 0.091
    # precision = tp / (tp + fp) = 0.5

    # - Low FPR can lead to high AUC of ROC curve, despite precision being
    # low! Therefore it is important to look at both metrics, along with
    # F-beta score

    ######################################################################
    # ROI on ad spend:

    # Same idea from prior: some cost c and return r

    # total_return = tp * r

    # total_spent = (tp + fp) * cost

    # roi = total_return / total_spent
    #     = (tp) / (tp + fp) * (r / cost)
    #     = precision * (r / cost)

    ######################################################################
    # Beginning model:

    # In this exercise, you will build an MLP classifier on the dataset of
    # images used in chapter 1. As a reminder, each image represents a number
    # 0 through 9 and the goal is to classify each imagine as a number. The
    # features used are specific pixel values ranging from 0-16 that make up
    # the image. After scaling features, you will evaluate the accuracy of the
    # classifier on the testing set.
    #
    # In your workspace, sample image data in DataFrame form is loaded as
    # image_data along with sklearn and pandas as pd. StandardScaler() from
    # sklearn.preprocessing is available as well.

    # Define X and y
    X, y = image_data.data, image_data.target

    # Scale features and split into training and testing
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=.2, random_state=0)

    # Create classifier, train and evaluate accuracy
    clf = MLPClassifier()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print(accuracy_score(y_test, y_pred))

    ######################################################################
    # MLPs for CTR:

    # In this exercise, you will evaluate both the accuracy score and AUC of
    # the ROC curve for a basic MLP model on the ad CTR dataset. Remember to
    # standardize the features before splitting into training and testing!
    #
    # X is available as the DataFrame with features, and y is available as a
    # DataFrame with target values. Both sklearn and pandas as pd are also
    # available in your workspace.

    # Scale features and split into training and testing
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0)

    # Create classifier and produce predictions
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=100)
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    # Get accuracy and AUC of ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Accuracy: %s" % (accuracy_score(y_test, y_pred)))
    print("ROC of AUC curve: %s" % (roc_auc))

    ######################################################################
    # Varying hyperparameters:

    # The number of iterations of training, and the size of hidden layers are
    # two primary hyperparameters that can be varied when working with a MLP
    # classifier. In this exercise, you will vary both separately and note how
    # performance in terms of accuracy and AUC of the ROC curve may vary.
    #
    # X_train, y_train, X_test, y_test are available in your workspace.
    # Features have already been standardized using a StandardScaler(). pandas
    # as pd, numpy as np are also available in your workspace.

    # Loop over various max_iter configurations
    max_iter_list = [10, 20, 30]
    for max_iter in max_iter_list:
        clf = MLPClassifier(hidden_layer_sizes=(4,),
                            max_iter=max_iter, random_state=0)
        # Extract relevant predictions
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
        y_pred = clf.fit(X_train, y_train).predict(X_test)

        # Get ROC curve metrics
        print("Accuracy for max_iter = %s: %s" % (
            max_iter, accuracy_score(y_test, y_pred)))
        print("AUC for max_iter = %s: %s" % (
            max_iter, roc_auc_score(y_test, y_score[:, 1])))

    # Create and loop over various hidden_layer_sizes configurations
    hidden_layer_sizes_list = [(4,), (8,), (16,)]
    for hidden_layer_sizes in hidden_layer_sizes_list:
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=10, random_state=0)
        # Extract relevant predictions
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
        y_pred = clf.fit(X_train, y_train).predict(X_test)

        # Get ROC curve metrics
        print("Accuracy for hidden_layer_sizes = %s: %s" % (
            hidden_layer_sizes, accuracy_score(y_test, y_pred)))
        print("AUC for hidden_layer_sizes = %s: %s" % (
            hidden_layer_sizes, roc_auc_score(y_test, y_score[:, 1])))

    ######################################################################
    # MLP Grid Search:

    # Hyperparameter tuning can be done by sklearn through providing various
    # input parameters, each of which can be encoded using various functions
    # from numpy. One method of tuning, which exhaustively looks at all
    # combinations of input hyperparameters specified via param_grid, is grid
    # search. In this exercise, you will use grid search to look over the
    # hyperparameters for a MLP classifier.
    #
    # X_train, y_train, X_test, y_test are available in your workspace, and
    # the features have already been standardized. pandas as pd, numpy as np,
    # are also available in your workspace.

    # Create list of hyperparameters
    max_iter = [10, 20]
    hidden_layer_sizes = [(8,), (16,)]
    param_grid = {'max_iter': max_iter,
                  'hidden_layer_sizes': hidden_layer_sizes}

    # Use Grid search CV to find best parameters using 4 jobs
    mlp = MLPClassifier()
    clf = GridSearchCV(estimator=mlp, param_grid=param_grid,
                       scoring='roc_auc', n_jobs=4)
    clf.fit(X_train, y_train)
    print("Best Score: ")
    print(clf.best_score_)
    print("Best Estimator: ")
    print(clf.best_estimator_)

    ######################################################################
    # F-beta score:

    # The F-beta score is a weighted harmonic mean between precision and
    # recall, and is used to weight precision and recall differently. It is
    # likely that one would care more about weighting precision over recall,
    # which can be done with a lower beta between 0 and 1. In this exercise,
    # you will calculate the precision and recall of an MLP classifier along
    # with the F-beta score using a beta = 0.5.
    #
    # X_train, y_train, X_test, y_test are available in your workspace, and
    # the features have already been standardized. pandas as pd and sklearn
    # are also available in your workspace. fbeta_score() from sklearn.metrics
    # is available as well.

    # Set up MLP classifier, train and predict
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=(16,),
                        max_iter=10, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    # Evaluate precision and recall
    prec = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    fbeta = fbeta_score(y_test, y_pred, beta=0.5, average='weighted')
    print(
        "Precision: %s, Recall: %s, F-beta score: %s" % (prec, recall, fbeta))

    ######################################################################
    # Precision, ROI, and AUC:

    # The return on investment (ROI) can be decomposed into the precision
    # multiplied by a ratio of return to cost. As discussed, it is possible
    # for the precision of a model to be low, even while AUC of the ROC curve
    # is high. If the precision is low, then the ROI will also be low. In this
    # exercise, you will use a MLP to compute a sample ROI assuming a fixed r,
    # the return on a click per number of impressions, and cost, the cost per
    # number of impressions, along with precision and AUC of ROC curve values
    # to check how the three values vary.
    #
    # X_train, y_train, X_test, y_test are available in your workspace, along
    # with clf as a MLP classifier, probability scores stored in y_score and
    # predicted targets in y_pred. pandas as pd and sklearn are also available
    # in your workspace.

    # Get precision and total ROI
    prec = precision_score(y_test, y_pred, average='weighted')
    r = 0.2
    cost = 0.05
    roi = prec * r / cost

    # Get AUC
    roc_auc = roc_auc_score(y_test, y_score[:, 1])

    print("Total ROI: %s, Precision: %s, AUC of ROC curve: %s" % (
        roi, prec, roc_auc))

    ######################################################################
    # Model comparison warmup:

    # In this exercise, you will run a basic comparison of the four categories
    # of outcomes between MLPs and Random Forests using a confusion matrix.
    # This is in preparation for an analysis of all the models we have
    # covered. Doing this warm-up exercise will allow you to compare and
    # contrast the implementation of these models and their evaluation for CTR
    # prediction.
    #
    # In the workspace, we have training and testing splits for X and y as
    # X_train, X_test for X and y_train, y_test for y respectively. Remember,
    # X contains our engineered features with user, device, and site details
    # whereas y contains the target (whether the ad was clicked). X has
    # already been scaled using a StandardScaler(). For future ad CTR
    # prediction models, the setup will be analogous.

    # Create the list of models in the order below
    names = ['Random Forest', 'Multi-Layer Perceptron']
    classifiers = [RandomForestClassifier(),
                   MLPClassifier(hidden_layer_sizes=(10,),
                                 max_iter=40)]

    # Produce a confusion matrix for all classifiers
    for name, classifier in zip(names, classifiers):
        print("Evaluating classifier: %s" % (name))
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

    ######################################################################
    # Evaluating precision and ROI:

    # In this exercise, you build upon the previous exercise and run an
    # MLPClassifier and compare it to three of the other classifiers run
    # earlier. For each classifier, you will compute the precision and implied
    # ROI on ad spend. As before, we have training and testing splits for X
    # and y as X_train, X_test for X and y_train, y_test for y respectively
    # and the features have already been standardized.

    # Create list of classifiers
    names = ['Logistic Regression', 'Decision Tree',
             'Random Forest', 'Multi-Layer Perceptron']
    clfs = [LogisticRegression(),
            DecisionTreeClassifier(), RandomForestClassifier(),
            MLPClassifier(hidden_layer_sizes=(5,), max_iter=50)]

    # Produce a classification report for all classifiers
    for name, classifier in zip(names, clfs):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        prec = precision_score(y_test, y_pred, average='weighted')
        r, cost = 0.2, 0.05
        roi = prec * r / cost
        print("ROI for %s: %s " % (name, roi))

    ######################################################################
    # Total scoring:

    # Remember that precision and recall might be weighted differently and
    # therefore the F-beta score is an important evaluation metric.
    # Additionally, the ROC of the AUC curve is an important complementary
    # metric to precision and recall since you saw prior how it may be the
    # case that a model might have a high AUC but low precision. In this
    # exercise, you will calculate the full set of evaluation metrics for each
    # classifier.
    #
    # A print_estimator_name() function is provided that will provide the name
    # for each classifier. X_train, y_train, X_test, y_test are available in
    # your workspace, and the features have already been standardized. pandas
    # as pd and sklearn are also available in your workspace.

    # Create classifiers
    clfs = [LogisticRegression(), DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(hidden_layer_sizes=(10,), max_iter=50)]

    # Produce all evaluation metrics for each classifier
    for clf in clfs:
        print("Evaluating classifier: %s" % (print_estimator_name(clf)))
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        prec = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        fbeta = fbeta_score(y_test, y_pred, beta=0.5, average='weighted')
        roc_auc = roc_auc_score(y_test, y_score[:, 1])
        print(
            "Precision: %s: Recall: %s, F-beta score: %s, AUC of ROC curve: %s"
            % (prec, recall, fbeta, roc_auc))

    ######################################################################
