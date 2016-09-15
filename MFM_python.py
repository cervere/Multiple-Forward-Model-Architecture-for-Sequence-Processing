
# coding: utf-8

# In[117]:

import numpy
import matplotlib


# In[151]:

def get_weights_k(weights, k, t, y, y_hat, eta):
    if t == 0:
        return numpy.random.uniform(0, 1, (3, 5))
    else:
        temp = eta * calc_lambda(k, t, delta)*y_hat[k]*(1-y_hat[k])
        return weights[k] + numpy.outer(temp, y)


# In[152]:

def sigmoid(x):
    sigmoid = 1/(1 + numpy.exp((-4 * x)))
    return sigmoid


# In[153]:

def seq_module(y, t, weights):
    y_hat = sigmoid(numpy.dot(weights, y))
    return (y_hat, t+1)


# Initializations

# In[154]:

weights = numpy.random.uniform(-1, 1, (4, 3, 5))
y = numpy.array([1, 0, 0, 0, 1])
t = 0
k = 4
y_desired = numpy.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
alpha = 0.5
delta = 4
eta = 0.2


# In[173]:

(y_hat, t) = seq_module(y, t, weights)
print y_hat


# In[174]:

def calc_lambda(k, t, delta):
    p = calc_p(k, t)
    p_k = 1
    p_k_total = 1
    p_total = 0
    for i in range(delta):
        p_k *= calc_p(k, t-i)
    for i in range(4):
        for j in range(delta):
            p_k_total *= calc_p(i, t-j) 
        p_total += p_k_total
    lambda_k = p_k/p_total
    return lambda_k


# In[175]:

def calc_p(k, t):
    if t <= 0:
        return 0.25
    else:
        return numpy.exp(-E(k, t)[1]/calc_sigma(alpha, t)**2)


# In[176]:

def E(k, t):
    error = y_desired[t-1] - y_hat[k]
    error_sum = (0.5) * (numpy.sum(error)**2)
    return (error, error_sum)


# In[177]:

def calc_sigma(alpha, t):
    error = list()
    sum_part = 0
    if t == 0:
        return 3
    else:
        for i in range(4):
            for j in range(4):
                error.append(E(j, t-1)[1])
            sum_part += min(error)            
        sigma = alpha*calc_sigma(alpha, t-1) + (1-alpha)*sum_part*0.25
        return sigma


# In[178]:

def response_selection(t, y_hat, y_desired):
    sum_part = 0
    for k in range(4):
        sum_part += y_hat[k]*calc_lambda(k, t, delta)
    return numpy.argmax(y_desired[t-1] + sum_part)


# In[179]:

response = response_selection(t, y_hat, y_desired)
y = numpy.array([1, 0, 0, 0, 0])
y[response + 1] = 1
print response


# In[180]:

weight_vec = list()
for i in range(k):
    weight_vec.append(get_weights_k(weights, i, t, y, y_hat, eta))


# In[181]:

weights = numpy.array(weight_vec)


# In[ ]:



