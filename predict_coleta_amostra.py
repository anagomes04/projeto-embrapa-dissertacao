import pandas as pd
import numpy as np
import math
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelResult:

    def __init__(self, real, pred):
        self.accuracy = accuracy_score(real, pred) * 100,
        self.precision = precision_score(real, pred, average='macro') * 100,
        self.recall = recall_score(real, pred, average='macro') * 100,
        self.f1_score = f1_score(real, pred, average='macro') * 100
        self.prediction = pred

    def __str__(self):
        return 'Accuracy: %.2f%%; Precision: %.2f%%; Recall: %.2f%%; F1-Score: %.2f%%' % (self.accuracy[0], self.precision[0], self.recall[0], self.f1_score)


def encode_variable(origin):
    encoder = LabelEncoder()
    return encoder.fit_transform(origin), encoder


def get_input_data():
    global database
    global encoders
    global expected_output

    input_columns = database.columns.values
    input_exceptions = [
        'DATA',
        'ANO',
        'control_index',
        expected_output
    ]

    for exception in input_exceptions:
        index = np.where(input_columns == exception)[0][0]
        input_columns = np.delete(input_columns, index)

    input_values = database[input_columns]

    columns_to_encode = ['INVASORA', 'TIPO DE FOLHA', 'PROFUNDIDADE (cm)', alternative]
    for column in columns_to_encode:
        encoded_value, encoder = encode_variable(input_values[column])
        input_values.loc[:, column] = encoded_value
        encoders[column] = encoder

    return input_values.values, input_columns


def get_output_data():
    global database
    global encoders
    global expected_output

    output_values = database[expected_output]

    encoded_value, encoder = encode_variable(output_values)
    output_values = encoded_value
    encoders[expected_output] = encoder

    return output_values


def shuffle():
    global input_data
    global output_data

    temp = list(zip(input_data, output_data))
    random.shuffle(temp)
    return zip(*temp)


def divide_sample(sample):
    breakpoint = math.floor(len(sample) * 0.7)
    return sample[:breakpoint], sample[breakpoint:]


def predict_with(model):
    global input_training, input_testing
    global output_training, output_testing

    model.fit(input_training, output_training)
    predictions = model.predict(input_testing)

    return model, ModelResult(output_testing, predictions)


def get_feature_importance(model):
    global input_columns

    feature_importance = model.feature_importances_
    feature_importance = feature_importance[:] * 100

    df = pd.DataFrame(feature_importance, input_columns)
    df = df.rename(columns={0: '%'})

    return df


def get_input_predict_analysis(result: ModelResult):
    global input_testing
    global input_columns
    global output_testing
    global encoders

    invasoras = np.array(np.stack(input_testing)[:, 0], dtype=np.uint8)
    invasoras = pd.Series(encoders['INVASORA'].inverse_transform(invasoras))
    lengthiest_invasora = invasoras.str.len().max()

    divisor_invasora = ' '.join(['' for _ in range(lengthiest_invasora - len('INVASORA') + 2)])

    predictions = result.prediction
    results = [pred == real for pred, real in zip(predictions, output_testing)]
    predictions = pd.Series(encoders['COLETA DA AMOSTRA'].inverse_transform(predictions))

    divisor_cultura = ' '.join(['' for _ in range(4)])

    analysis = f'INVASORA{divisor_invasora}COLETA DA AMOSTRA{divisor_cultura}PREDIÇÃO'

    for invasora, predicted, result in zip(invasoras, predictions, results):
        divisor_invasora = ' '.join(['' for _ in range(lengthiest_invasora - len(invasora) + 2)])
        divisor_coleta_amostra = ' '.join(['' for _ in range(len('COLETA DA AMOSTRA') - len(predicted) + 4)])
        analysis += f'\n{invasora}{divisor_invasora}{predicted}{divisor_coleta_amostra}{result}'

    return analysis


if __name__ == '__main__':
    database = pd.read_csv('data/base_unificada_sem_label.csv')
    expected_output = 'COLETA DA AMOSTRA'
    alternative = 'PLANTAÇÃO'
    encoders = {}

    input_data, input_columns = get_input_data()
    output_data = get_output_data()

    input_scaler = StandardScaler()
    input_data = input_scaler.fit_transform(input_data)

    input_data, output_data = shuffle()

    input_training, input_testing = divide_sample(input_data)
    output_training, output_testing = divide_sample(output_data)

    decision_tree, decision_tree_result = predict_with(DecisionTreeClassifier())
    random_forest, random_forest_result = predict_with(RandomForestClassifier())
    knn, knn_result = predict_with(KNeighborsClassifier(n_neighbors=2))
    svm, svm_result = predict_with(SVC(kernel='linear'))

    print('RESULTADOS ANÁLISE COM PREDIZINDO:', expected_output)
    print('Decision tree:\t', decision_tree_result)
    print('Random forest:\t', random_forest_result)
    print('KNN:\t\t\t', knn_result)
    print('SVM:\t\t\t', svm_result)

    input_testing = input_scaler.inverse_transform(input_testing)

    with open('result/decision_tree_fi_coleta_amostra.txt', 'w') as f:
        f.write(f'{get_feature_importance(decision_tree)}')

    with open('result/random_forest_fi_coleta_amostra.txt', 'w') as f:
        f.write(f'{get_feature_importance(random_forest)}')

    with open('result/decision_tree_analise_predicao_coleta_amostra.txt', 'w') as f:
        f.write(get_input_predict_analysis(decision_tree_result))

    with open('result/random_forest_analise_predicao_coleta_amostra.txt', 'w') as f:
        f.write(get_input_predict_analysis(random_forest_result))

    with open('result/knn_analise_predicao_coleta_amostra.txt', 'w') as f:
        f.write(get_input_predict_analysis(knn_result))

    with open('result/svm_analise_predicao_coleta_amostra.txt', 'w') as f:
        f.write(get_input_predict_analysis(svm_result))

