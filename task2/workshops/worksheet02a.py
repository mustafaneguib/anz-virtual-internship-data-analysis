
# coding: utf-8

# # COMP90051 Workshop 2
# ## Part A: Linear regression
# 
# ***
# 
# Our aim for this part of the workshop is to fit a linear model from scratch—relying only on the `numpy` library. We'll experiment with two implementations: one based on iterative updates (coordinate descent) and another based on linear algebra. Finally, to check the correctness of our implementation, we'll compare its output to the output of `sklearn`.
# 
# Firstly we will import the relevant libraries (`numpy`, `matplotlib`, etc.).

# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import io


# To check what a command does simply type `object?`. For example:

# In[4]:


get_ipython().run_line_magic('pinfo', 'np.arange')


# ### 1. Review
# In lectures, we saw that a linear model can be expressed as:
# $$y = w_0 + \sum_{j = 1}^{m} w_j x_j = \mathbf{w} \cdot \mathbf{x} $$
# where 
# 
# * $y$ is the *target variable*;
# * $\mathbf{x} = [x_1, \ldots, x_m]$ is a vector of *features* (we define $x_0 = 1$); and
# * $\mathbf{w} = [w_0, \ldots, w_m]$ are the *weights*.
# 
# To fit the model, we *minimise* the empirical risk with respect to $\vec{w}$. In this simplest case (square loss), this amounts to minimising the sum of squared residuals:
# 
# $$SSR(\mathbf{w}) = \sum_{i=1}^{n}(y_i - \mathbf{w} \cdot \mathbf{x}_i)^2$$
# 
# **Note:** For simplicity, we'll consider the case $m = 1$ (i.e. only one feature excluding the intercept).

# ### 2. Data set
# We'll be working with some data from the Olympics—the gold medal race times for marathon winners from 1896 to 2012. The code block below reads the data into a numpy array of floats, and prints the result.

# In[15]:


# CSV file with variables YEAR,TIME
csv = """1896,4.47083333333333
1900,4.46472925981123
1904,5.22208333333333
1908,4.1546786744085
1912,3.90331674958541
1920,3.5695126705653
1924,3.8245447722874
1928,3.62483706600308
1932,3.59284275388079
1936,3.53880791562981
1948,3.6701030927835
1952,3.39029110874116
1956,3.43642611683849
1960,3.2058300746534
1964,3.13275664573212
1968,3.32819844373346
1972,3.13583757949204
1976,3.07895880238575
1980,3.10581822490816
1984,3.06552909112454
1988,3.09357348817
1992,3.16111703598373
1996,3.14255243512264
2000,3.08527866650867
2004,3.1026582928467
2008,2.99877552632618
2012,3.03392977050993"""

# Read into a numpy array (as floats)
olympics = np.genfromtxt(io.BytesIO(csv.encode()), delimiter=",")
print(olympics)
print(len(olympics))


# We'll take the race time as the *target variable* $y$ and the year of the race as the only non-trivial *feature* $x = x_1$.

# In[16]:


x = olympics[:, 0:1]
y = olympics[:, 1:2]

print(olympics[:,0:1])


# Plotting $y$ vs $x$, we see that a linear model could be a decent fit for this data.

# In[17]:


plt.plot(x, y, 'rx')
plt.ylabel("y (Race time)")
plt.xlabel("x (Year of race)")
plt.show()


# ### 3. Iterative solution (coordinate descent)

# Expanding out the sum of square residuals for this simple case (where $\mathbf{w}=[w_0, w_1]$) we have:
# $$SSR(w_0, w_1) = \sum_{i=1}^{n}(y_i - w_0 - w_1 x_i)^2$$
# Let's start with an initial guess for the slope $w_1$ (which is clearly negative from the plot).

# In[18]:


w1 = -0.4


# Then using the maximum likelihood update, we get the following estimate for the intercept $w_0$:
# $$w_0 = \frac{\sum_{i=1}^{n}(y_i-w_1 x_i)}{n}$$

# In[19]:


def update_w0(x, y, w1):
    return  (y-x*w1).mean() #we need to normalize so we are taking the mean

w0 = update_w0(x, y, w1)
print(w0)


# Similarly, we can update $w_1$ based on this new estimate of $w_0$:
# $$w_1 = \frac{\sum_{i=1}^{n} (y_i - w_0) \times x_i}{\sum_{i=1}^{n} x_i^2}$$

# In[25]:


def update_w1(x, y, w0):
    return (((y-w0)*x)/(x*x))

w1 = update_w1(x, y, w0)
print(w1)


# Let's examine the quality of fit for these values for the weights $w_0$ and $w_1$. We create a vector of "test" values `x_test` and a function to compute the predictions according to the model.

# In[26]:


x_test = np.arange(1890, 2020)[:, None]

def predict(x_test, w0, w1): 
    return ... # fill in


# Now plot the test predictions with a blue line on the same plot as the data.

# In[27]:


def plot_fit(x_test, y_test, x, y): 
    plt.plot(x_test, y_test, 'b-')
    plt.plot(x, y, 'rx')
    plt.ylabel("y (Race time)")
    plt.xlabel("x (Year of race)")
    plt.show()

plot_fit(x_test, predict(x_test, w0, w1), x, y)


# We'll compute the sum of square residuals $SSR(w_0,w_1)$ on the training set to measure the goodness of fit.

# In[ ]:


def compute_SSR(x, y, w0, w1): 
    return ... # fill in

print(compute_SSR(x, y, w0, w1))


# It's obvious from the plot that the fit isn't very good. 
# We must repeat the alternating parameter updates many times before the algorithm converges to the optimal weights.

# In[ ]:


for i in np.arange(10000):
    w1 = update_w1(x, y, w0) 
    w0 = update_w0(x, y, w1) 
    if i % 500 == 0:
        print("Iteration #{}: SSR = {}".format(i, compute_SSR(x, y, w0, w1)))
print("Final estimates: w0 = {}; w1 = {}".format(w0, w1))


# Let's try plotting the result again.

# In[ ]:


plot_fit(x_test, predict(x_test, w0, w1), x, y)


# Does more than 10 iterations considerably improve fit in this case?

# ### 4. Linear algebra solution

# In lectures, we saw that it's possible to solve for the optimal weights $\mathbf{w}^\star$ analytically. The solution is
# $$\mathbf{w}^* = \left[\mathbf{X}^\top \mathbf{X}\right]^{-1} \mathbf{X}^\top \mathbf{y}$$
# where
# $$\mathbf{X} = \begin{pmatrix} 
#         1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n 
#     \end{pmatrix} 
#   \quad \text{and} \quad 
#   \mathbf{y} = \begin{pmatrix} 
#           y_1 \\ y_2 \\ \vdots \\ y_n
#       \end{pmatrix}
# $$
# 
# We construct $\mathbf{X}$ in the code block below, remembering to include the $x_0 = 1$ column for the bias (intercept).

# In[ ]:


X = np.hstack((np.ones_like(x), x))
print(X)


# Although we can express $\mathbf{w}^\star$ explicitly in terms of the matrix inverse $(\mathbf{X}^\top \mathbf{X})^{-1}$, this isn't an efficient way to compute $\mathbf{w}$ numerically. It is better instead to solve the following system of linear equations:
# $$\mathbf{X}^\top\mathbf{X} \mathbf{w}^\star = \mathbf{X}^\top\mathbf{y}$$
# 
# This can be done in numpy using the command

# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.linalg.solve')


# which gives

# In[ ]:


w = ... # fill in
print(w)


# Plotting this solution, as before:

# In[ ]:


w0, w1 = w
plot_fit(x_test, predict(x_test, w0, w1), x, y)


# You should verify that the sum of squared residuals $SSR(w_0, w_1)$, match or beats the earlier iterative result.

# In[ ]:


print(compute_SSR(x, y, w0, w1))


# **Note:** The error we computed above is the *training* error. It doesn't assess the model's generalization ability, it only assesses how well it's performing on the given training data. In later worksheets we'll assess the generalization ability of models using held-out evaluation data.

# ### 5. Solving using scikit-learn

# Now that you have a good understanding of what's going on under the hood, you can use the functionality in `sklearn` to solve linear regression problems you encounter in the future. Using the `LinearRegression` module, fitting a linear regression model becomes a one-liner as shown below.

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(x, y)


# The `LinearRegression` module provides access to the bias weight $w_0$ under the `intercept_` property

# In[ ]:


lr.intercept_


# and the non-bias weights under the `coef_` property

# In[ ]:


lr.coef_


# You should check that these results match the solution you obtained previously. Note that sklearn also uses a numerical linear algebra solver under the hood.
