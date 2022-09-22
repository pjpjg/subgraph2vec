import os
import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, f1_score, matthews_corrcoef, precision_score,\
    recall_score, average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import sem
import pandas as pd
import time

# -----------------------------------------------------------------

METRICS = {'gmean': ['gmean', 'sensitivity', 'specificity'],
           'ave_precision': ['ave_precision']}
N_SPLITS = 10
SVM_PARAM_GRID = [{'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4], 'kernel': ['linear']},
                  {'C': [1, 10, 0.1, 100, 0.01, 1000, 0.001, 1e4, 1e-4],
                   'gamma': [1, 0.5, 3, 0.2, 10, 0.1, 0.03, 0.01, 0.001, 1e-4], 'kernel': ['rbf']}]
GRID_SPLITS = 10
MODEL = ['svm']

# -----------------------------------------------------------------

def gmean_score(y_test, results_test):
    tn, fp, fn, tp = confusion_matrix(y_test, results_test).ravel()
    tpr = float(tp) / (float(tp) + float(fn))
    tnr = float(tn) / (float(tn) + float(fp))
    return np.sqrt(tpr * tnr)


def grid_search_metrics(metric, model, param_grid, grid_splits, X_train, y_train):
    if metric == 'gmean':
        clf = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring=make_scorer(gmean_score, greater_is_better=True), cv=grid_splits,
                           verbose=0, n_jobs=-1)
  
    elif metric == 'ave_precision':
        clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='average_precision', cv=grid_splits,
                           verbose=0, n_jobs=-1)
    else:
        print('Metric not implemented')
        return
    clf.fit(X_train, y_train)
    return clf.best_params_


def prep_exp(experiment, input_data, embeddings):
    # load input dataset
    input_data_filepath = os.path.join(os.getcwd(), 'input', input_data)
    with open(input_data_filepath, 'r') as data:
        data_matrix = [list(x.split(",")) for x in data]
    data.close()
    for go in data_matrix:
        if go[-1] == '0\n':
            go[-1] = '0'
        elif go[-1] == '1\n':
            go[-1] = '1'
    # load embeddings
    emb_file = '{}.emb'.format(embeddings)
    embeddings_filepath = os.path.join(os.getcwd(), emb_file)
    with open(embeddings_filepath, 'r') as goVec:
        goVec_matrix = [list(x.split(' ')) for x in goVec]
    goVec.close()
    goVec_dict = {}
    if embeddings.find('anc') != -1:
        start = 0
    else: start = 1
    for i in range(start, len(goVec_matrix)):          
        go_list = goVec_matrix[i]
        if go_list[-1] == '\n':
            go_list.pop()
        goVec_dict[go_list[0]] = go_list[1:]
    # prepare experiment datasets
    features, class_labels = [], []
    
    if experiment == 'binary':
        for i_1 in range(1, len(data_matrix[0])):
            vector = []
            for i_2 in range(1, len(data_matrix)-1):
                if data_matrix[i_2][i_1] == '1':
                    vector.append(float(1))
                else:
                    vector.append(float(0))
            features.append(vector)
            class_labels.append(float(data_matrix[-1][i_1]))
        features = np.array(features)
        class_labels = np.array(class_labels)
        
    elif experiment == 'both':
        for i_1 in range(1, len(data_matrix[0])):
            binary_vector, emb_vector = [], []
            for i_2 in range(1, len(data_matrix)-1):
                if data_matrix[i_2][i_1] == '1':
                    binary_vector.append(float(1))
                    if data_matrix[i_2][0] != 'class':
                        GO_ID = data_matrix[i_2][0]
                        go_emb = list(np.float_(goVec_dict[GO_ID]))
                        emb_vector.append(go_emb)
                else:
                    binary_vector.append(float(0))
            processed_emb_vector = np.mean(emb_vector, axis=0)
            # processed_emb_vector = np.sum(emb_vector, axis=0)
            binary_vector = np.array(binary_vector)
            combined_vector = np.concatenate((processed_emb_vector, binary_vector))
            features.append(combined_vector)
            class_labels.append(float(data_matrix[-1][i_1]))
        features = np.array(features)
        class_labels = np.array(class_labels)
        
    elif experiment == 'emb':
        for i_1 in range(1, len(data_matrix[0])):
            vector = []
            for i_2 in range(1, len(data_matrix)-1):
                if data_matrix[i_2][i_1] == '1':
                    if data_matrix[i_2][0] != 'class':
                        GO_ID = data_matrix[i_2][0]
                        go_emb = list(np.float_(goVec_dict[GO_ID]))
                        vector.append(go_emb)
            processed_vector = np.average(vector, axis=0)
            # processed_vector = np.sum(vector, axis=0)
            features.append(processed_vector)
            class_labels.append(float(data_matrix[-1][i_1]))
        features = np.array(features)
        # scaler = MinMaxScaler()
        # features = scaler.fit_transform(features)
        class_labels = np.array(class_labels)
        
    features = np.asarray(features, dtype=np.float32)
    class_labels = np.asarray(class_labels, dtype=np.float32)
    return features, class_labels

# -----------------------------------------------------------------

def run_experiment(experiment, input_data, embeddings, metrics=METRICS, n_splits=N_SPLITS, svm_param_grid=SVM_PARAM_GRID,
                   grid_splits=GRID_SPLITS):

    np.random.seed(7)
    features, class_labels = prep_exp(experiment, input_data, embeddings)

    # build dataframe for results
    dataset_name = input_data[:len(input_data)-4]
    all_metrics = []
    for each_main_metric in metrics.keys():
        all_metrics.extend(metrics[each_main_metric])
    header = pd.MultiIndex.from_product([MODEL, all_metrics], names=['models', 'metrics'])
    df = pd.DataFrame(index=[dataset_name], columns=header)
    
    for grid_search_metric in metrics.keys():

        print(grid_search_metric)

        # create another dictionary to hold fold scores for each grid_search_metrics, create lists for auprc
        metric_fold_scores = {}
        for metric in metrics[grid_search_metric]:
            metric_fold_scores[metric] = []
        auc_y_real = []
        auc_y_prob = []

        # set up k-fold for cv by reading in preDefinedSplit
        ps_filepath = os.path.join(os.getcwd(), 'preDef_splits', '{}.csv'.format(input_data[:-4]))
        ps_file = np.loadtxt(ps_filepath)
        ps = PredefinedSplit(ps_file)
            
        # set up k-fold for cv for each model
        # skf = StratifiedKFold(n_splits=n_splits)         # cen: random_state=7, shuffle=True
        # for train_index, test_index in ps.split(features, class_labels):
        
        fold = 1
        
        for train_index, test_index in ps.split():
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = class_labels[train_index], class_labels[test_index]

            # timing each fold
            begin = time.time()

            # perform svm
            svc = SVC(cache_size=1000, random_state=7)   # an attempt at reproducbility in GridSearchCV     
            params = grid_search_metrics(grid_search_metric, svc, svm_param_grid, grid_splits, X_train, y_train)
            kernel_value = params['kernel']
            c_value = float(params['C'])
            if kernel_value == 'rbf':
                gamma_value = float(params['gamma'])
                svc_trained = SVC(kernel=kernel_value, C=c_value, gamma=gamma_value)
                svc_trained_prob = SVC(kernel=kernel_value, C=c_value, gamma=gamma_value, probability=True)
            else:
                svc_trained = SVC(kernel=kernel_value, C=c_value)
                svc_trained_prob = SVC(kernel=kernel_value, C=c_value, probability=True)
            svc_trained.fit(X_train, y_train)
            svc_trained_prob.fit(X_train, y_train)
            results_test = svc_trained.predict(X_test)
            results_test_prob = svc_trained_prob.predict_proba(X_test)

            # get fold scores
            if grid_search_metric == 'gmean':
                tn, fp, fn, tp = confusion_matrix(y_test, results_test).ravel()
                tpr, tnr = float(tp) / (float(tp) + float(fn)), float(tn) / (float(tn) + float(fp))
                gmean = np.sqrt(tpr * tnr)
                metric_fold_scores['gmean'].append(gmean)
                metric_fold_scores['sensitivity'].append(tpr)
                metric_fold_scores['specificity'].append(tnr)
            elif grid_search_metric == 'ave_precision':
                ave_precision = average_precision_score(y_test, results_test_prob[:, 1])
                metric_fold_scores['ave_precision'].append(ave_precision)
                auc_y_real.append(y_test)
                auc_y_prob.append(results_test_prob[:, 1])
            else:
                print('Metric not implemented.')

            # timing each fold
            total_time = time.time() - begin

            print('{} {} fold {}'.format(fold, grid_search_metric, dataset_name))
            print('Running scores: {}'.format(metric_fold_scores[grid_search_metric]))
            print('Time taken: {}'.format(total_time))

            fold += 1

        # evaluate metric
        for metric in metrics[grid_search_metric]:
            average = np.average(metric_fold_scores[metric])
            se = sem(metric_fold_scores[metric])

            # populate df with results
            df.loc[dataset_name, ('svm', metric)] = [average, se]

    auc_y_real = np.concatenate(auc_y_real)
    auc_y_prob = np.concatenate(auc_y_prob)

    np.savetxt(os.path.join('output', 'probs', '{}_real.txt'.format(dataset_name)),
               auc_y_real)
    np.savetxt(os.path.join('output', 'probs', '{}_proba.txt'.format(dataset_name)),
               auc_y_prob)     

    # return df as a csv file
    filename = '{}.csv'.format(dataset_name)
    filepath = os.path.join('output', filename)
    df.to_csv(filepath)

# -----------------------------------------------------------------

DATASETS = ['DM_BP', 'DM_CC', 'DM_MF', 'DM_BP-CC', 'DM_BP-MF', 'DM_CC-MF', 'DM_BP-CC-MF',
            'MM_BP', 'MM_CC', 'MM_MF', 'MM_BP-CC', 'MM_BP-MF', 'MM_CC-MF', 'MM_BP-CC-MF',
            'SC_BP', 'SC_CC', 'SC_MF', 'SC_BP-CC', 'SC_BP-MF', 'SC_CC-MF', 'SC_BP-CC-MF',
            'CE_BP', 'CE_CC', 'CE_MF', 'CE_BP-CC', 'CE_BP-MF', 'CE_CC-MF', 'CE_BP-CC-MF']

DATASET = 'bp'
EXPERIMENT = 'both'
EMBEDDINGS = 'anc2vec_embeddings_128'

if __name__ == '__main__':
    for dataset in DATASETS:
        print(dataset)
        dataset = '{}.txt'.format(dataset)
        run_experiment(EXPERIMENT, dataset, EMBEDDINGS)
        print('----------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------')

