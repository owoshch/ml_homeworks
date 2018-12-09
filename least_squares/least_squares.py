import numpy as np
import tqdm

def normal_equation(X, y):
    return np.dot(np.linalg.pinv(X), y)


def linear_prediction(X, w):
    return np.dot(X, w)

def mserror(y, y_pred):
    return np.sum((y - y_pred) ** 2) / len(y)

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad = (w[0] * X[train_ind, 0]  + w[1] * X[train_ind, 1] + w[2] * X[train_ind, 2] + w[3] * X[train_ind, 3] - y[train_ind])
    grad0 = X[train_ind, 0] * grad
    grad1 = X[train_ind, 1] * grad
    grad2 = X[train_ind, 2] * grad
    grad3 = X[train_ind, 3] * grad
    return  np.array(w) - (2. / X.shape[0]) * eta * np.array([grad0, grad1, grad2, grad3])


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e5,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    
    weights_history = []
    """
    means = X.mean(axis = 0)
    stds = X.std(axis = 0)
    X = (X - means) / stds
    """
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
        
    # Сюда будем записывать ошибки на каждой итерации
    errors = []

    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)
        
    # Основной цикл
    #while weight_dist > min_weight_dist and iter_num < max_iter:
    for iter_num in tqdm.tqdm(range(int(max_iter)), total=max_iter):
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        
        random_ind = np.random.randint(X.shape[0])
        w0 = w

        w = stochastic_gradient_step(X, y, w, random_ind, eta)

        errors.append(mserror(y, np.dot(X, w)))
        
        weights_history.append(w)
        
    return np.array(w), np.array(errors), np.array(weights_history)